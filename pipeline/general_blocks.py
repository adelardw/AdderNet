import torch
import torch.nn as nn
from typing import Optional
from pipeline.masked_conv import *
from pipeline.mobile_block import MobileBlock

def list_fn(num_blocks, params):
    """

    Args:
        num_blocks (int): the number of base blocks in encoder or decoder
        params (Any): the parameters you need to wrap in a list up to the length of num_blocks 

    Returns:
        List[Any]: list of params the length of num_blocks

    """
    if isinstance(params, list):
        if len(params) < num_blocks:
            params += [params[-1]] * (num_blocks - len(params))
        else:
            params = params[:num_blocks]
        
        return params
    else:
        return list_fn(num_blocks, [params])


class BaseBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, inverted: bool = False,
                act_func: nn.Module = nn.SiLU(), do_bn: bool = True, dp: float | int = 0.4):
        """

        Args:
            in_channels (int): input channels of encoder or decoder
            out_channels (int): out channels of encoder or decoder
            kernel_size (int): kernel size for convolution
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
            self.conv_inverted = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,padding = pad)
        else:
            #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = pad)
            self.conv = OutMaskedConv2d(in_channels, out_channels, kernel_size, padding = pad)
        self.nb =  nn.BatchNorm2d(out_channels) if do_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dp) if dp > 0 else nn.Identity()
        self.act = act_func




class AdaptiveResBlock(nn.Module):
    def __init__(self, in_channels: int,
                        out_channels: int,
                        kernel_size: int,
                        use_mobile: bool = False, 
                        downscaling: Optional[bool] = False,
                        inverted: bool = False,
                        act_func: nn.Module = nn.SiLU(),
                        do_bn: bool = True,
                        do_sc: bool = True,
                        dp: float | int = 0.4):
        """

        Args:
            in_channels (int): input channels of encoder or decoder
            out_channels (int): out channels of encoder or decoder
            kernel_size (int): kernel size for convolution
            use_mobile (bool): if True: use MobileBLock (see mobile_blocks module), else: use BaseBlocks. Defaults to False.
            downscaling (optional[bool]): if False: use MaxPool2d for downscaling. If False: use TransposeConv2d for upscaling. 
                                          if None: dont use any scaling. Defaults True.
            inverted (bool): if False: this is encoder mode - convolutions are used.
                             if True: this is dencoder mode - transpose convolutions are used.
                             Defaults to False.

            do_sc (bool): if True: using skip connection
                          if False: dont using skip connection. Defaults to True.

            act_func (nn.Module): chosen activation function. Defaults to nn.SiLU().
            do_bn (bool): if true: do BatchNormalization, else Identuty. Defaults to True.
            dp (float | int): dropout probability. Defaults to 0.4.
        """

        super().__init__()

        self.do_sc = do_sc

        if use_mobile:
            self.block = MobileBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       act_func=act_func,
                                       inverted=inverted,
                                       dp=dp,
                                       do_bn=do_bn)
        else:
            self.block = BaseBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       act_func=act_func,
                                       dp=dp,
                                       inverted=inverted,
                                       do_bn=do_bn)
        if not downscaling:
            self.scale = nn.ConvTranspose2d(out_channels, out_channels, stride=2,
                                            kernel_size=2,padding=0, output_padding=0) #nn.Upsample(scale_factor=2)
        elif downscaling:
            self.scale = nn.MaxPool2d(kernel_size=2)
        else:
            self.scale = nn.Identity()

        if self.do_sc:

            
            self.adapt_res = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, bias=False) if in_channels != out_channels else \
                                 nn.Identity()
            
        
    def forward(self, x):
        if self.do_sc:
            return self.scale(self.block(x) + self.adapt_res(x))
        else:
            return self.scale(self.block(x))