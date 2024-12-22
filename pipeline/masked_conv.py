import torch
import torch.nn as nn

class OutMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple | int, *args, **kwargs):

        """
        General: Masked convolution output using down triangular mask.
        Args:
            in_channels (int): num input channels
            out_channels (int): num out channels
            kernel_size (tuple | int): convolution kernel size
        """
        super().__init__(in_channels, out_channels, kernel_size,*args, **kwargs)

    def forward(self, x):
        output = super().forward(x)
        mask = torch.tril(torch.ones_like(output)).to(output.device)
        return output * mask


class WeightMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple | int, *args, **kwargs):

        """
        General: Masked convolution weights using down triangular mask.
        Args:
            in_channels (int): num input channels
            out_channels (int): num out channels
            kernel_size (tuple | int): convolution kernel size
        """
        
        super().__init__(in_channels, out_channels, kernel_size,*args, **kwargs)
        
        self.weight.data =  self.weight.data * torch.tril(torch.ones_like(self.weight.data))

class OutWeightMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple | int, *args, **kwargs):

        """
        General: Masked convolution weights and masked convolution output using down triangular masks.
        Args:
            in_channels (int): num input channels
            out_channels (int): num out channels
            kernel_size (tuple | int): convolution kernel size
        
        """
        super().__init__(in_channels, out_channels, kernel_size,*args, **kwargs)
        
        self.weight.data =  self.weight.data * torch.tril(torch.ones_like(self.weight.data))

    def forward(self, x):
        output = super().forward(x)
        mask = torch.tril(torch.ones_like(output)).to(output.device)
        return output * mask
    
