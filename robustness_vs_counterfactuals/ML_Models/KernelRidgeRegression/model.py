import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

def squared_exp_kernel(X, Y):
    """
        A squared exponential kernel function. See KernelRidgeRegression.__init__ for an interface description.
    """
    dmat = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)  # shape [M, N]
    return torch.exp(-0.5*dmat.pow(2))


"""
    Kernel Ridge Regression
"""

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel_func, lambd: float = 10.0, num_of_classes: int = 1):
        """
            kernel_func: a vectorized version of the kernel function for points of dimension [input_dim]
            For inputs X (Shape[M, input_dim]) and Y (Shape [N, input_dim]), kernel func should return a
            [M, N]-matrix with the kernel evaluated between the pairs of points. This matrix is positive definite for M=N.
        """
        super().__init__()
        self.num_of_classes = num_of_classes
        self.kernel = kernel_func
        self.internal_weights = 0
        self.points = 0
        self.lambd = lambd

    def fit(self, X: torch.tensor, y: torch.tensor):
        """ 
            Fit the kernel ridge model to data X with labels Y. We have
            y^hat = k(x^pred, X)*(k(X,X)+lambda*I)^-1y
            In this function we caluclate the term (k(X,X)+lambda*I)^-1y, which we store as internal_weights, i.e.,
            internal weights = k(X,X)+lambda*I)^-1y
        """
        self.points = X
        kernel_mat = self.kernel(self.points, self.points)
        # For efficient computation we rearrange to (k(X,X)+lambda*I)*internal_weights = y.
        self.internal_weights = torch.linalg.solve(kernel_mat + self.lambd*torch.eye(len(X)), y.reshape(-1, 1)).flatten()
        print("kernel weights shape:", self.internal_weights.shape)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return torch.sigmoid(self.predict_with_logits(x))

    def predict_with_logits(self, x: torch.tensor) -> torch.tensor:
        k_premult = self.kernel(x, self.points)
        return torch.matmul(k_premult, self.internal_weights.reshape(-1, 1)).flatten()

    def predict_from_weights(self, x: torch.tensor, weights: torch.tensor):
        """
            Predict logits with different weights, but using the data points stored with this model.
            x: (B, input_dim) inputs. 
            w: weights of (B, len(self.points))
        """
        k_premult = self.kernel(x, self.points)
        return torch.matmul(k_premult, weights.reshape(-1, 1)).flatten()

    def jac_kx(self, x_cf):
        """
            Compute Jacobian of the kernel function k_X(x).
            x_cf: Batched Input points [B, input_dim]
            return: Batched jacobian [B, N, input_dim] of self.kernel(x_cf, self.points) w.r.t x_cf
        """
        x_cfgrad = x_cf.clone().requires_grad_()
        func = lambda x: torch.sum(self.kernel(x, self.points), dim=0)  # output shape N (self.npoints)
        ret = jacobian(func, x_cfgrad,)

        print('Jacobian shape:', ret.shape)
        return ret.transpose(0, 1)

    def compute_weights_under_removal(self, y: torch.tensor):
        """
            Compute model weight change.
            To accomplish that we use information from equation 9 in "Rethinking Influence Functions of Neural Networks
            in the Over-parameterized Regime" by Zhang and Zhang (2021) (https://arxiv.org/pdf/2112.08297.pdf).
            Column n represents the change of the internal_weights under removal of point n.
        """
        kernel_mat = (self.kernel(self.points, self.points) + self.lambd*torch.eye(len(self.points)))
        with torch.no_grad():
            kernel_mat_inv = torch.linalg.inv(kernel_mat) 
        kity = kernel_mat_inv.matmul(y.reshape(-1, 1))/torch.diag(kernel_mat_inv).reshape(-1, 1)
        diff = -kernel_mat_inv*kity.reshape(1, -1)
        return diff


