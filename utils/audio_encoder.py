import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d_model=256, out_dim=512):
        super().__init__()
        self.input = nn.Conv1d(1, d_model, kernel_size=1)
        self.mamba = Mamba(d_model=d_model, d_state=d_model // 2, d_conv=7)
        self.group_norm = nn.GroupNorm(4, d_model)
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(d_model, out_dim, kernel_size=7, padding='same'),
            nn.Softmax(dim=-1)  
        )
    
    def forward(self, x):
        residual = x 
        x = self.input(x)
        out = self.mamba(x.permute(0, -1, 1)).permute(0, -1, 1)
        out = self.group_norm(out + x)
        out = self.spatial_attention(out) 
        return F.max_pool1d(out + residual, kernel_size=2)  
    
class Conv1dBlock(nn.Module):
    def __init__(self, out_channels=512, kernel_size=3, out_dim=512, num_groups=32):
        super().__init__()
        self.input = nn.Conv1d(1, out_channels, kernel_size=kernel_size, padding='same', dilation=3)
        
        
        self.conv_dilation = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size, padding='same', dilation=3
        )
        self.activation = nn.Sequential(
            nn.ELU(),
            nn.Dropout1d(0.5),
            nn.GroupNorm(4, out_channels)
        )
        
        
        self.depthwise = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=1, groups=num_groups
        )
        self.pointwise = nn.Conv1d(
            out_channels, out_dim, 
            kernel_size=kernel_size, padding='same'
        )
        

        self.spatial_attention = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding='same'),
            nn.Softmax(dim=-1) 
        )
    
    def forward(self, x):
        residual = x
        x = self.input(x)
        
        out = self.conv_dilation(x)
        out = self.activation(out + x)  
        
        
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.activation(out) + out
        
        attention = self.spatial_attention(out)
        
        return F.max_pool1d(attention + residual, kernel_size=2)  
    

class AudioEncoder(nn.Module):
    def __init__(self, d_model=8, out_dim=256, kernel_size=3, num_groups=8,
                 out_size = (4, 4)):
        super().__init__()

        self.mamba_block = MambaBlock(d_model,out_dim)
        self.conv_block = Conv1dBlock(out_dim, kernel_size, out_dim, num_groups)
        self.out = nn.Sequential(nn.Conv1d(out_dim * 2, out_dim, kernel_size=1),
                                  nn.MaxPool1d(kernel_size=4))
    
        self.avg_pooler = nn.AdaptiveAvgPool2d(out_size)
    
    def forward(self, x):

        out = self.out(torch.cat([self.mamba_block(x), self.conv_block(x)], dim=1)).unsqueeze(-1)
        out = self.avg_pooler(out)
        return out 