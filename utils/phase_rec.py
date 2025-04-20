import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
    


class MambaPhaseCorrector(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.conv_in = nn.Conv2d(2, d_model, kernel_size=3, padding=1)
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=4,  
            d_conv=4,    
            expand=2             )
        self.conv_out = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, mag, phase):
        x = torch.cat([mag, phase], dim=1)  # [B, 2, F, T]
        x = self.conv_in(x)  # [B, d_model, F, T]
        
        
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)  # [B*F, T, d_model]
        x = self.mamba(x)  
        x = x.reshape(B, F, T, C).permute(0, 3, 1, 2)  # [B, C, F, T]
        
        return phase + torch.tanh(self.conv_out(x))  # Residual + коррекция
    

class EnhancedPhaseCorrector(nn.Module):
    def __init__(self, num_channels=32, num_layers=5, dilation_groups=[1,2,4,8]):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=(5,7), padding='same'),
            nn.GroupNorm(4, num_channels),
            nn.ELU()
        )
        

        self.dilation_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_groups[i % len(dilation_groups)]
            self.dilation_blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, 
                             kernel_size=(3,5), 
                             padding='same',
                             dilation=(dilation, 1)),
                    nn.GroupNorm(4, num_channels),
                    nn.ELU(),
                    nn.Dropout2d(0.1)
                )
            )
        

        self.freq_attention = nn.Sequential(
            nn.Conv2d(num_channels, num_channels//4, kernel_size=(1,1)),
            nn.ELU(),
            nn.Conv2d(num_channels//4, num_channels, kernel_size=(1,1)),
            nn.Sigmoid()
        )
        

        self.output_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Conv2d(num_channels, 1, kernel_size=3, padding='same'),
            nn.Tanh()
        )
        
    def forward(self, mag, noisy_phase):

        x = torch.cat([mag, noisy_phase], dim=1)
        

        x = self.initial_conv(x)
        
        residual = x
        for block in self.dilation_blocks:
            x = block(x) + residual
            residual = x
        

        attn = self.freq_attention(x)
        x = x * attn

        phase_diff = self.output_conv(x)

        return noisy_phase + 0.5 * phase_diff
    
class CompatiblePhaseCorrector(nn.Module):
    def __init__(self, base_channels=16):  
        super().__init__()
        

        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=(3,5), padding='same'),  
            nn.GroupNorm(max(1, base_channels//4), base_channels),  
            nn.ELU()
        )
        

        self.dilation_blocks = nn.Sequential(
            *[self._make_block(base_channels, d) for d in [1, 2, 1, 4]] 
        )
        

        self.freq_attention = nn.Sequential(
            nn.Conv2d(base_channels, 4, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(4, base_channels, kernel_size=1),
            nn.Sigmoid()
        )
        

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding='same'),
            nn.Tanh()
        )
    
    def _make_block(self, channels, dilation):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding='same', dilation=(dilation,1)),
            nn.GroupNorm(max(1, channels//4), channels),
            nn.ELU(),
            nn.Dropout2d(0.05)
        )

    def forward(self, mag, noisy_phase):

        x = torch.cat([mag, noisy_phase], dim=1)
        x = self.initial_conv(x)
        

        residual = x
        for block in self.dilation_blocks:
            x = block(x) + residual
            residual = x
        

        x = x * self.freq_attention(x)
        
        return noisy_phase + 0.3 * self.output_conv(x)