import lightning as L
import torchaudio
from loaders import *
from pipeline import *
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.optim as optim
import torch.nn.functional as F
import librosa
import torch
import numpy as np


class SpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.loss_fn = nn.MSELoss()
        self.spectral_loss = SpectralConvergengeLoss()
        self.audio_len = audio_len
        
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.model.parameters(), lr=3.41e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7, eta_min=1e-6)
        
        lr_scheduler_config = {
            "scheduler": scheduler,

            "interval": "epoch",

            "frequency": 1,
        }

        return [optimizer], [lr_scheduler_config]
    
    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            mode="max",
            save_top_k=1,
            filename="best-checkpoint-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            every_n_epochs=1
        )
        return [checkpoint_callback]
    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    
    @staticmethod
    def spec_pad(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) and spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
        
    def forward(self, mixed_waveforms, speech_waveforms):

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        magnitude_mixed = torch.abs(stft_mixed)
        magnitude_clean = torch.abs(stft_clean)
        
        output = self.model(magnitude_mixed)
        output_padded = self.spec_pad(output, stft_clean)

        return output_padded, magnitude_clean
    
    def run(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
        if mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.unsqueeze(0)
        if mixed_waveforms.ndim == 1:
            mixed_waveforms = mixed_waveforms.unsqueeze(0).unsqueeze(0)
        
        
        mixed_waveforms = mixed_waveforms[..., :self.audio_len].to(self.device)
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        phase = torch.angle(stft_mixed).detach().cpu()
        magnitude = torch.abs(stft_mixed)
        output = self.model(magnitude).detach().cpu()

        spec_padded = self.spec_pad(output, phase)
        spec_cleaned = (spec_padded * torch.exp(1j * phase)).numpy()
        
        cleaned = librosa.istft(spec_cleaned, n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center)
        
        
        return cleaned
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        output, stft_clean = self.forward(mixed_waveforms, speech_waveforms)
        
        loss = self.loss_fn(output, stft_clean) + self.spectral_loss(output, stft_clean)
        
        metrics = {
            f"{kind}_loss": loss,
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True)

        return loss
    

