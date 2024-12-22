import torch
import torch.nn as nn

class SiSDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        """

        Args:
            eps (float): eps for stabibiluty calculations. Defaults to 1e-9.
        """
        super().__init__()
        self.eps = eps


    def forward(self, output, target):

        alpha = torch.sum(output * target, dim=-1,keepdim=True) / torch.norm(target, dim=-1)**2 

        proj = alpha * target

        proj_norm = torch.norm(proj, dim=-1)
        diff_norm = torch.norm((proj - output), dim=-1)

        return  -(10 * (torch.log10(proj_norm**2 / (diff_norm**2 + self.eps )))).mean()


class CIRMLoss(nn.Module):
    def __init__(self, k: int | float = 10, c: float  = 0.1):
        """

        Args:
            k (int | float): constant of compressions mask in interval [-k, k]. Defaults to 10.
            c (float): constant of controls mask steepness. Defaults to 0.1.
        """
        super().__init__()

        self.k = k
        self.c = c
    
    def forward(self, x):
        return self.k * nn.functional.tanh(self.c* x / 2)