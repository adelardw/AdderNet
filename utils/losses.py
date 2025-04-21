import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class SiSDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        """

        Args:
            eps (float): eps for stabibiluty calculations. Defaults to 1e-9.
        """
        super().__init__()
        self.eps = eps


    def forward(self, output, target):

        alpha = torch.sum(output * target, dim=-1,keepdim=True) / (torch.norm(target, dim=-1)**2 + self.eps)

        proj = alpha * target

        proj_norm = torch.norm(proj, dim=-1)
        diff_norm = torch.norm((proj - output), dim=-1)

        return  -(10 * (torch.log10(proj_norm**2 / (diff_norm**2 + self.eps )))).mean()



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
        
    def forward(self, output, target):
        
        log_trc = torch.log(torch.abs(target) + 1)
        log_out = torch.log(torch.abs(output) + 1)
        return F.l1_loss(log_out, log_trc)

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

class MultiResolutionLoss(nn.Module):
    def __init__(self, n_ffts: list = [256, 512, 1025, 2096]):
        super().__init__()
        self.n_ffts = n_ffts
        self.transforms = nn.ModuleList([
            T.Spectrogram(
                n_fft=n_fft,
                power=1.0,
                normalized=False,
                center=False
            )
            for n_fft in n_ffts])
        
    def forward(self, clean, enhanced):
        losses = []
        for transform in self.transforms:
            transform.window = transform.window.to(clean.device)
            
            S_clean = transform(clean)
            S_enhanced = transform(enhanced)
            losses.append(F.l1_loss(S_clean, S_enhanced))
            
        return torch.mean(torch.stack(losses))

class STFTLoss(nn.Module):

    def forward(self, stft_true, stft_pred):        
        return torch.mean(torch.abs(stft_true - stft_pred))

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        self.component_mse = ComponentMSE()
    def forward(self, anchor, pos, neg):

        return F.relu(self.component_mse(anchor, pos) - self.component_mse(anchor, neg) + self.margin).mean()   