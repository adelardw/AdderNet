import torch
import torch.nn as nn
from typing import Optional
from pipeline.masked_conv import *


class MobileBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                        inverted: bool = False,
                        act_func: nn.Module = nn.SiLU(),
                        do_bn: bool = True,
                        dp: float | int = 0.4):
        """

        Args:
            in_channels (int): input channels of encoder or decoder
            out_channels (int): out channels of encoder or decoder
            kernel_size (int): kernel size
            inverted (bool) : If False: this is encoder mode - convolutions are used.
                              If True: this is dencoder mode - transpose convolutions are used.
                              Defaults to False.

            act_func (nn.Module): Chosen activation function. Defaults to nn.SiLU().
            do_bn (bool): If true: do Batch Normalization, else Identuty. Defaults to True.
            dp (float | int): Dropout probability. Defaults to 0.4.
        """
        super().__init__()
    

        assert kernel_size % 2 != 0, 'Kernel must be odd'
            
          
        pad = kernel_size // 2  
        if inverted:
            self.conv_inverted = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=kernel_size,padding=pad)
            
            self.depth_wise_inverted = nn.ConvTranspose2d(out_channels, out_channels,groups=out_channels,
                                          kernel_size=kernel_size)
                                          
            self.point_wise_inverted = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1,padding=pad)
        else:
            
            self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,padding=pad)
            
            self.depth_wise = nn.Conv2d(out_channels, out_channels,groups=out_channels,
                                          kernel_size=kernel_size)
                                          
            #self.point_wise = nn.Conv2d(out_channels, out_channels, kernel_size=1,padding=pad)
            self.point_wise = nn.OutMaskedConv2d(out_channels, out_channels, kernel_size=1,padding=pad)
        
        self.nb =  nn.BatchNorm2d(out_channels) if do_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dp) if dp > 0 else nn.Identity()
        self.act = act_func