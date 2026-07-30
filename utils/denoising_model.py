import torch
import torch.nn as nn
from utils.general_blocks import *
from utils.encoder import *
from utils.decoder import *
from utils.bottlenecks import *
from utils.audio_encoder import *
import torchaudio
from typing import Any
from copy import deepcopy
import torch.nn.functional as F

class DenoisingModel(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func =  'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),

                        input_shape: tuple = (512, 512),
                        hidden_gru: int = 2048,
                        num_gru_cells: int = 2,
                        dp_gru: float = 0.3):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

            input_signal_size (int): num signal pts: sample rate * num signal seconds.Defaults to 80000.
            n_fft (int): num pts for FT. Defaults to 512.
            hop_length (int): Window step. Defaults to 128.
            center (bool): if center == True - use padding reflect mode for calc stft. Defaults to True.

            hidden_gru (int): num of gru hidden neurons. Defaults to 512.
            num_gru_cells (int): num of gru cells. Defaults to 2.
            dp_gru (float): dropout for gru each gru cells. Defaults to 0.3.
        """
        super().__init__()


        assert encoder_parameters['out_channels'][-1] == decoder_parameters['in_channels']
        
    
        num_encoder_blocks = encoder_parameters['num_blocks']
        features_scaling = 2**num_encoder_blocks
        scaled_frequnecy = input_shape[0] // features_scaling

        self.encoder = SpectrumEncoder(**encoder_parameters)
        
        self.gru = nn.GRU(input_size=encoder_parameters['out_channels'][-1] * scaled_frequnecy, 
                          hidden_size=hidden_gru,
                          batch_first=True,
                          dropout=dp_gru,
                          num_layers=num_gru_cells,
                          bias=False,
                          bidirectional=True)
        
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(2 * hidden_gru, encoder_parameters['out_channels'][-1]* scaled_frequnecy, bias=False)
        self.tanh_gru = nn.Tanh()

        self.decoder = SpectrumDecoder(**decoder_parameters)

    def forward(self, x):

        encoded = self.encoder(x)

        batch_size, channels, frequency, time = encoded.shape
        input_size = frequency * channels

        x = encoded.permute(0, 3, 2, 1)

        x = x.reshape(batch_size, time, input_size)
        
        gru_out, hidden = self.gru(x)
        out = self.tanh(gru_out)
        out = self.linear(out)
        out = self.tanh_gru(out)
        decoder_input = out.view(batch_size, channels,frequency, time)

        decoded = self.decoder(decoder_input)

        return decoded
    
    

class DenoisingModelV2(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3)):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

        """
        super().__init__()


        assert encoder_parameters['out_channels'][-1] == decoder_parameters['in_channels']
        
        self.encoder = SpectrumEncoder(**encoder_parameters)
        self.act = nn.Tanh()
        self.decoder = SpectrumDecoder(**decoder_parameters)

    def forward(self, x):

        x = self.encoder(x)
        x = self.act(x)
        return self.decoder(x)
    
    


class DenoisingModelUnet(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3)):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

        """
        super().__init__()


        #assert encoder_parameters['out_channels'][-1] == decoder_parameters['in_channels']
        
        self.encoder = SpectrumEncoder(**encoder_parameters)
        self.act = nn.Tanh()
        self.decoder = SpectrumDecoder(**decoder_parameters)

    def forward(self, x):

        sc = []
        encoder_features = self.encoder.encoder_features
        decoder_features = self.decoder.decoder_features

        for module in encoder_features.children():
            x = module(x)
            sc.append(x)
        
        x = self.act(x)
        sc = sc[::-1]
        for module, skp in zip(decoder_features.children(), sc):
            x = module(x + skp)  

        return x


class DenoisingModelComplexUnet_v1(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3)):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

        """
        super().__init__()
        encoder_parameters = deepcopy(encoder_parameters) 
        decoder_parameters = deepcopy(decoder_parameters)
        
        encoder_parameters['in_channels'] = 2 * encoder_parameters['in_channels'] 
        decoder_parameters['out_channels'][-1] = 2 * decoder_parameters['out_channels'][-1] 

        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                        decoder_parameters=decoder_parameters)
        
        

    def forward(self, x):
        
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        x_concat = torch.cat([magnitude, phase], dim=1)
        out = self.model(x_concat)
        out_magnitude, out_phase = torch.chunk(out, 2, dim=1)
        
        return out_magnitude, out_phase
    


class DenoisingModelComplexUnet_v2(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3)):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

        """
        super().__init__()
        encoder_parameters = deepcopy(encoder_parameters) 
        decoder_parameters = deepcopy(decoder_parameters)
        
        encoder_parameters['in_channels'] = 2 * encoder_parameters['in_channels'] 
        decoder_parameters['out_channels'][-1] = 2 * decoder_parameters['out_channels'][-1] 

        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                        decoder_parameters=decoder_parameters)
        
        

    def forward(self, x):
        
        x_concat = torch.cat([x.real, x.imag], dim=1)
        out = self.model(x_concat)
        out_real, out_imag = torch.chunk(out, 2, dim=1)
        
        return out_real, out_imag
        
    

class DenoisingModelUnetBottleneckWithAudioEncoder(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = 'elu',
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                        audio_encoder_parameters: dict = dict(d_model = 8,
                                                                out_dim= 256,
                                                                kernel_size = 3,
                                                                num_groups = 8,
                                                                out_size = (4, 4))):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = 'elu',
                                                                                    do_bn = True,
                                                                                    do_sc = True, 
                                                                                    dp = 0.4,

        """
        super().__init__()


        #assert encoder_parameters['out_channels'][-1] == decoder_parameters['in_channels']
        
        self.encoder = SpectrumEncoder(**encoder_parameters)
        self.audio_encoder = AudioEncoder(**audio_encoder_parameters)
        self.bottleneck = SpeechEnhancementBottleneck(audio_channels=audio_encoder_parameters['out_dim'],
                                                      unet_channels=encoder_parameters['out_channels'][-1])
        self.decoder = SpectrumDecoder(**decoder_parameters)

    def forward(self, mag, x):

        sc = []
        #encoder_features = self.encoder.encoder_features
        #decoder_features = self.decoder.decoder_features

        for module in self.encoder.encoder_features.children():
            mag = module(mag)
            sc.append(mag)
        
        x = self.audio_encoder(x)
        mag = self.bottleneck(x, mag)
        sc = sc[::-1]
        for module, skp in zip(self.decoder.decoder_features.children(), sc):
            mag = module(mag + skp)  

        return mag