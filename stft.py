import torch
from typing import Optional

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
        stft_amplitude = torch.stft(mono_channel, n_fft, hop_length, win_length, window,   return_complex=True)
        answ.append(stft_amplitude)
    
    return torch.stack(answ, dim=1)