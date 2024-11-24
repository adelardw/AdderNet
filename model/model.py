import torch
import torch.nn as nn



class BaseBlock(nn.Sequential):
    def __init__(self, in_channels,
                       out_channels,
                       act_function,
                       )