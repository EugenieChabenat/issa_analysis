import torch
import torch.nn as nn
import argparse

from models import BarlowTwins

def get_parser():
    parser = argparse.ArgumentParser(description='Model-specific parameters')
    parser.add_argument('--queue-size', default=8192, type=int,
                        help='number of images stored in queue for computing factorization score')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='explained variance ratio threshold for image subspace')
    parser.add_argument('--pos-weight', default=0.03, type=float,
                        help='cam pos targets will be multiplied by this number')
    parser.add_argument('--scale-weight', default=20.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    parser.add_argument('--color-weight', default=15.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    parser.add_argument('--equivariant-weight', default=1.0, type=float,
                        help='equivariant loss will be multiplied by this number')
    parser.add_argument('--factorization-weight', default=0.2, type=float,
                        help='factorization loss will be multiplied by this number')
    return parser

class Model(BarlowTwins):
    def __init__(self, args):
        super().__init__(args)
        
        sizes = {'bn': torch.Size([self.projector_sizes[-1]]), 'avgpool': torch.Size((2048,))}
        
        # create embedding matrix
        self.register_buffer("embedding", nn.init.orthogonal_(torch.empty(7, *sizes['bn'])))
        
        # factorization
        self.queue_size = args.queue_size
        self.register_buffer("queue", torch.randn(*sizes['avgpool'], self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.pca = PCA(threshold=args.threshold)
        
    def forward(self, y1, y2, **kwargs):
        # args and kwargs should be on gpu
        delta_pos = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_pos_x', 'cam_pos_y']], dim=1)*self.args.pos_weight
        delta_scale = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_scale']], dim=1)*self.args.scale_weight
        delta_color = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['brightness', 'contrast', 'saturation', 'hue']], dim=1)*self.args.color_weight
        is_not_bw = ((1.0-kwargs['applied_RandomGrayscale_0'])*(1.0-kwargs['applied_RandomGrayscale_1'])).squeeze()
        is_color_jittered = (kwargs['applied_ColorJitter_0']*kwargs['applied_ColorJitter_1']).squeeze()
        delta_color = torch.einsum('b,bm->bm',is_not_bw*is_color_jittered,delta_color)
        
        delta_vec = torch.cat([delta_pos, delta_scale, delta_color], dim=1).matmul(self.embedding)
        x1 = self.backbone(y1)
        z1 = self.bn(self.projector(x1))
        z1 = z1 + delta_vec
        x2 = self.backbone(y2)
        z2 = self.bn(self.projector(x2))
        
        # factorization
        with torch.no_grad():
            self.pca.fit(self.queue.t()) # fact_queue.t() - (queue_size, n_features)
        acts = torch.stack([x1,x2], dim=0)
        acts_centered = (acts - acts.mean(dim=0)).reshape(-1,acts.size()[-1]) # (2*batch_size,n_features)
        acts_centered_proj = self.pca.transform(acts_centered)
        factorization_loss = acts_centered_proj.var(dim=0).sum()/acts_centered.var(dim=0).sum()
        self._dequeue_and_enqueue(acts.mean(dim=0))
        
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        equivariant_loss = on_diag + self.args.lambd * off_diag
        
        # get total loss
        loss_components = {
            'Equivariant Loss': equivariant_loss.item(),
            'Factorization Loss': factorization_loss.item(),
        }
        loss = equivariant_loss*self.args.equivariant_weight + factorization_loss*self.args.factorization_weight
        return loss, loss_components
    
    def get_encoder(self):
        return self.backbone
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.queue_size % batch_size == 0:  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr
        else:
            print("Incompatible batch, skipping queue update")
    
### Utilities ###

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class PCA():
    def __init__(self, threshold=0.9):
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold
        
    def fit(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/decomposition/_pca.py
        """
        assert X.ndim == 2
        n_samples, n_features = X.size()[0], X.size()[1]
        
        self.mean_ = X.mean(dim=0)
        X = X - self.mean_
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        components_ = Vt
        
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        
        ratio_cumsum = torch.cumsum(explained_variance_ratio_, dim=0)
        n_components = torch.searchsorted(ratio_cumsum, self.threshold, right=True) + 1
        
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        
    def transform(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/decomposition/_base.py
        """
        X = X - self.mean_
        return X.mm(self.components_.t())
    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
