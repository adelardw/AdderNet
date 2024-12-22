import torch
import torch.nn as nn
from pipeline.general_blocks import *
from pipeline.encoder import *
from pipeline.decoder import *

def get_stft_out_size(N, n_fft = 512,  hop_length = 128, center = True):
    """

    Args:
        N (int): num signal pts
        n_fft (int): num pts for FT. Defaults to 512.
        hop_length (int): Window step. Defaults to 128.
        center (bool): if center == True - use padding reflect mode for calc stft. Defaults to True.

    Returns:
        tuple: Tuple of F, T
    """
    if center:
        N_padded = N + 2 * (n_fft // 2)
        T = 1 + (N_padded - n_fft) // hop_length
        F = 1 + n_fft // 2

    
    else:
        T = 1 + (N - n_fft) // hop_length
        F = 1 + n_fft // 2


    return F, T

class DenoisingModel(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = nn.SiLU(),
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = False, 
                                                        act_func = nn.SiLU(),
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                        input_signal_size: int = 80000,
                        n_fft: int = 512,
                        hop_length: int = 128,
                        center: bool = True,
                        hidden_gru: int = 2048,
                        num_gru_cells: int = 2,
                        dp_gru = 0.3):
        """
        General: See Encoder and Decoder Models for understanding
        Args:
            encoder_parameters (dict): Dict of Encoder Parameters. Defaults to dict(in_channels=3,
                                                                                    out_channels = [64, 96, 128],
                                                                                    kernel_sizes = [3, 5, 7],
                                                                                    use_mobile = False,
                                                                                    act_func = nn.SiLU(),
                                                                                    do_bn = True,
                                                                                    do_sc = True,
                                                                                    dp = 0.4, 
                                                                                    num_blocks = 3).

            decoder_parameters (dict): Dict of Decoder Parameters. Defaults to dict(in_channels=128, 
                                                                                    out_channels = [96, 64, 3],
                                                                                    kernel_sizes = [3, 5, 7], 
                                                                                    use_mobile = False,
                                                                                    act_func = nn.SiLU(),
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
        
        input_stft_size = get_stft_out_size(N=input_signal_size, n_fft=n_fft,hop_length=hop_length, center=center)
    
        num_encoder_blocks = encoder_parameters['num_blocks']
        features_scaling = 2**num_encoder_blocks
        scaled_frequnecy = input_stft_size[0] // features_scaling

        self.encoder = SpectrumEncoder(**encoder_parameters)
        
        
        self.gru = nn.GRU(input_size=encoder_parameters['out_channels'][-1] * scaled_frequnecy, 
                          hidden_size=hidden_gru,
                          batch_first=True,
                          dropout=dp_gru,
                          num_layers=num_gru_cells,
                          bias=False)
        
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_gru, encoder_parameters['out_channels'][-1]* scaled_frequnecy, bias=False)
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
        decoder_input = out.view(batch_size, channels,frequency, time)

        decoded = self.decoder(decoder_input)

        return decoded