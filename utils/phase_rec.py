import torch
import torch.nn as nn

class PhaseCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1,),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()  
        )
        
    def forward(self, mag, noisy_phase):
        phase_diff = self.conv(torch.cat([mag, noisy_phase], dim=1))
        return noisy_phase + phase_diff
    

class PhaseCorrectorDilation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3,5), padding='same'),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(3,3), padding='same', dilation=(2,1)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding='same'),
            nn.Tanh()
        )
        
    def forward(self, mag, noisy_phase):
        phase_diff = self.conv(torch.cat([mag, noisy_phase], dim=1))
        return noisy_phase + phase_diff