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
from torchmetrics.audio import( ScaleInvariantSignalDistortionRatio as SISDR, SignalDistortionRatio as SDR,
                                SignalNoiseRatio as SNR, ScaleInvariantSignalNoiseRatio as SISNR)


class SpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.loss_fn = nn.MSELoss()
        self.spectral_loss = SpectralConvergengeLoss()
        self.audio_len = audio_len
        self.metric = dict(snratio = SNR(),
                            sdratio = SDR(),
                            sisdratio = SISDR(),
                            sisnratio = SISNR())
        
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
        checkpoint_callback_1 = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            save_top_k=3,
            filename="best-checkpoint-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            every_n_epochs=1
        )
        
        checkpoint_callback_2 = ModelCheckpoint(
            monitor="valid_snr",
            mode="max",
            save_top_k=3,
            filename="best-checkpoint-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            every_n_epochs=1
        )
        
        return [checkpoint_callback_1, checkpoint_callback_2]
    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    
    @staticmethod
    def pad(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) and spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    

    def padd_forward(self, mixed_waveforms):

        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        magnitude_mixed = torch.abs(stft_mixed)
        
        output = self.model(magnitude_mixed)
        output_padded = self.pad(output, stft_mixed)

        return output_padded
    
    def compute_metrics(self,cleaned_wf, clean_wf):
        cleaned_wf = cleaned_wf.detach().cpu()
        clean_wf = clean_wf.detach().cpu()
        cleaned_wf_shape = cleaned_wf.shape[-1]
        clean_wf_shape = clean_wf.shape[-1]
        if cleaned_wf.shape[1] != 1:
            cleaned_wf = cleaned_wf.sum(1, keepdims=True)
        if clean_wf_shape == min(clean_wf_shape, cleaned_wf_shape):
            cleaned_wf = cleaned_wf[:, :, :clean_wf_shape]
        else:
            clean_wf = clean_wf[:, :, :cleaned_wf_shape]

        self.metric['snratio'].update(cleaned_wf, clean_wf)
        self.metric['sisdratio'].update(cleaned_wf, clean_wf)
        self.metric['sdratio'].update(cleaned_wf, clean_wf)
        self.metric['sisnratio'].update(cleaned_wf, clean_wf)
     
    
    def forward(self, mixed_waveforms):
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
        if mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.unsqueeze(0)
        if mixed_waveforms.ndim == 1:
            mixed_waveforms = mixed_waveforms.unsqueeze(0).unsqueeze(0)
        
        if mixed_waveforms.shape[-1] >= self.audio_len:
            mixed_waveforms = mixed_waveforms[..., :self.audio_len].to(self.device)
        else:
            audio_shape = mixed_waveforms.shape[-1]
            need2pad = self.audio_len - audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0,need2pad)).to(self.device) 
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        phase = torch.angle(stft_mixed).detach().cpu() 
        magnitude = torch.abs(stft_mixed)
        with torch.no_grad():
            output = self.model(magnitude).detach().cpu()                                                                                                                         
        magn_padded = self.pad(output, phase)
       
        return magn_padded, phase
    
    def run(self, mixed_waveforms):
        
        out, phase = self.forward(mixed_waveforms)
        cleaned_stft = (out * torch.exp(1j * phase)).numpy()
        cleaned = librosa.istft(cleaned_stft, n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center)
        
    
        return cleaned
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        stft_clean = self.stft(speech_waveforms).to(self.device)
        clean_magnitude = torch.abs(stft_clean)
        
        output = self.padd_forward(mixed_waveforms)
        
        loss = self.loss_fn(output, clean_magnitude) + self.spectral_loss(output, clean_magnitude)
        
        cleaned = torch.tensor(self.run(mixed_waveforms))
        self.compute_metrics(cleaned, speech_waveforms)
    
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

    def on_train_epoch_end(self):
        snratio = self.metric['snratio'].compute()
        sisdratio = self.metric['sisdratio'].compute()
        sdratio = self.metric['sdratio'].compute()
        sisnratio = self.metric['sisnratio'].compute()
        
        metrics = dict(train_snr = snratio,
                       train_sisdr = sisdratio,
                       train_sdr = sdratio,
                       train_sisnr = sisnratio)
        
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True)
    
    def on_validation_epoch_end(self):
        snratio = self.metric['snratio'].compute()
        sisdratio = self.metric['sisdratio'].compute()
        sdratio = self.metric['sdratio'].compute()
        sisnratio = self.metric['sisnratio'].compute()
        
        metrics = dict(valid_snr = snratio,
               valid_sisdr = sisdratio,
               valid_sdr = sdratio,
               valid_sisnr = sisnratio)
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True)
    

