import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SDR(nn.Module):

    def __init__(self, eps = 1e-12):
        super().__init__()
        self.eps = eps
    
    def forward(self, x, y, y_hat):

        alpha = torch.norm(y, dim=-1, keepdim=True)**2 / (torch.norm(y, dim=-1, keepdim=True)**2 + torch.norm(x - y, dim=-1, keepdim=True)**2 + self.eps)
        loss_p1 = -alpha * (torch.sum(y * y_hat, dim=-1) / (torch.norm(y, dim=-1, keepdim=True) * torch.norm(y_hat, dim=-1, keepdim=True) + self.eps))
        loss_p2 = -(torch.ones_like(alpha) - alpha) * (torch.sum((x - y) * (x - y_hat), dim=-1) / (torch.norm((x - y), dim=-1, keepdim=True) * torch.norm((x - y_hat), dim=-1, keepdim=True) + self.eps)) 

        return (loss_p1 + loss_p2).mean()
    


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


class SpectralConvergengeLoss(nn.Module):

    def forward(self, output, target):

        return torch.norm(target - output, p="fro") / torch.norm(target, p="fro")

class ComponentMSE(nn.Module):
    
    def forward(self, output, target):
        
        return F.mse_loss(output.real, target.real) + F.mse_loss(output.imag, target.imag)

class CircularMSE(nn.Module):
    def forward(self, output, target):
        
        return torch.min(F.mse_loss(output, target), 2*torch.pi - F.mse_loss(output, target))
    

class AntiWrappingLoss(nn.Module):
    """
    Args:
        See https://arxiv.org/pdf/2305.13686 formula (9) and next sequences
    """    
        
    def forward(self, output, target):
        delta = target - output
        calc = torch.abs(delta - 2*torch.pi*torch.round(delta / (2 * torch.pi)))
        return calc.mean()
    
class LogMagnitudeLoss(nn.Module):
    def __init__(self, eps= 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, output, target):
        
        log_trc = torch.log(target + self.eps)
        log_out = torch.log(output + self.eps)
        return F.mse_loss(log_out, log_trc)

class PhaseSensetiveLoss(nn.Module):
    def forward(self, rec_mag, clean_mag, out_phase, clean_phase):
        
        return torch.mean((clean_mag - rec_mag*torch.cos(clean_phase - out_phase))**2)
    
class PhaseLoss(nn.Module):
    
    def forward(self, clean_phase, out_phase):
        return torch.mean((torch.cos(clean_phase - out_phase) - 1)**2)
    
class GroupDelayLoss(nn.Module):
    def forward(self, phase, target_phase):
        grad_phase = torch.diff(phase, dim=2)
        grad_target = torch.diff(target_phase, dim=2)
        return F.l1_loss(grad_phase, grad_target)
