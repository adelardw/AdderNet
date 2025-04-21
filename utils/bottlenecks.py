import torch.nn as nn
import torch

class SpeechEnhancementBottleneck(nn.Module):
    def __init__(self, audio_channels=256, unet_channels=256):
        super().__init__()

        self.audio_proj = nn.Conv2d(audio_channels, unet_channels, kernel_size=1)
        self.unet_proj = nn.Conv2d(unet_channels, unet_channels, kernel_size=1)
        
        
        self.noise_suppression_att = nn.Sequential(
            nn.Conv2d(unet_channels * 2, unet_channels // 8, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(unet_channels // 8, unet_channels * 2, kernel_size=1),
            nn.Sigmoid()  
        )
        
       
        self.speech_filter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(unet_channels, unet_channels),
        )
        

        self.final_conv = nn.Conv2d(2*unet_channels, unet_channels, kernel_size=3, padding='same')

    def forward(self, audio_feats, unet_feats):

        audio_proj = self.audio_proj(audio_feats)  
        unet_proj = self.unet_proj(unet_feats)     

        combined = torch.cat([audio_proj, unet_proj], dim=1)  

        attn_weights = self.noise_suppression_att(combined)    
        filtered = combined * attn_weights
        out = self.final_conv(filtered)

        return out
