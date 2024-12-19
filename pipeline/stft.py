import torch
from typing import Optional
import torch.nn.functional as F


def resize_tensor(input_tensor, target_tensor):
    """
    Расширяет или обрезает input_tensor до размера target_tensor.
    
    Args:
        input_tensor (torch.Tensor): 
        target_tensor (torch.Tensor): 
    
    Returns:
        torch.Tensor: Тензор с размером target_tensor.
    """
    target_size = target_tensor.shape
    
    return F.interpolate(input_tensor, size=target_size[-2:])


def stft_multichannel_reshape(audio: torch.Tensor, n_fft: int, hop_length: int, win_length: Optional[int]=None,
                              window: Optional[torch.Tensor]=None):
    """

    Args:
        audio (torch.Tensor): audio sample shapes like [batch size, time]
        n_fft (int): num fft points
        hop_length (int): step between windows
        win_length (Optional[int] ): window lenght. Defaults to None.
        window (Optional[torch.Tensor] ): chosen window. Defaults to None.

    Returns:
        torch.Tensor: STFT Amplitude
    """
    batch_size, channels, time = audio.shape
    audio_reshaped = audio.reshape(-1, time)

    stft_amplitude = torch.stft(audio_reshaped, n_fft, hop_length, win_length, window,   return_complex=True)
    _, frequencies, time = stft_amplitude.shape
    stft_amplitude = stft_amplitude.reshape(batch_size, channels, frequencies, time)
    return stft_amplitude 



def stft_multichannel(audio: torch.Tensor, n_fft: int, hop_length: int, win_length: Optional[int]=None,
                              window: Optional[torch.Tensor]=None):
    """
    Args:
        audio (torch.Tensor): audio sample shapes like [batch size, time]
        n_fft (int): num fft points
        hop_length (int): step between windows
        win_length (Optional[int] ): window lenght. Defaults to None.
        window (Optional[torch.Tensor] ): chosen window. Defaults to None.

    Returns:
        torch.Tensor: STFT Amplitude
    """
    
    
    batch_size, channels, time = audio.shape
    audio_reshaped = audio.reshape(-1, time)
    answ = []
    for c in  range(channels):
        mono_channel = audio[:, c, :]
        stft = torch.stft(mono_channel, n_fft, hop_length, win_length, window,   return_complex=True)
        answ.append(stft)
    
    return torch.stack(answ, dim=1)


def masked_istft_multichannel(outs, inputs, n_fft=512, hop_length=128, center = True):
    
    b, c, f, t = outs.shape
    
    outs = resize_tensor(outs, inputs)

    masked = outs * inputs
    
    answ = []
    for ch in range(c):
        channel = masked[:, ch, ...]

        complex_channel = torch.complex(channel, torch.zeros_like(channel))
        istft_c = torch.istft(complex_channel, n_fft=n_fft, hop_length=hop_length, center=center, return_complex=False)
        answ.append(istft_c)

    return torch.stack(answ, dim=1)