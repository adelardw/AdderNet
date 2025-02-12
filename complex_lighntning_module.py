from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
import torch.optim as optim
import torch.nn.functional as F
import librosa
import lightning as L
import torchaudio
from loaders import *
from pipeline import *




class SpectrogramLightningModelComplexUnet_V1(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelComplexUnet_v1(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.mse_lfn = nn.MSELoss()
        self.sc_lfn = SpectralConvergengeLoss()
        self.phase_lfn = nn.MSELoss()
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
        early_stop_callback = EarlyStopping(
                            monitor='valid_loss',
                            min_delta=0.00,
                            patience=20,
                            verbose=True,
                            mode='min'
                        )

        checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{valid_loss:.2f}',
            save_top_k=3,
            verbose=True,
            monitor='valid_loss',
            mode='min',
            save_last=True,  
            every_n_epochs=1,  
            auto_insert_metric_name=True)
        
        return [early_stop_callback, checkpoint_callback]

    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    
    @staticmethod
    def spec_pad(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) or spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
        
    def forward(self, mixed_waveforms, speech_waveforms):

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        trc_magnitude = torch.abs(stft_clean)
        trc_phase = torch.angle(stft_clean)
        
        out_magnitude, out_phase = self.model(stft_mixed)
        out_magnitude = self.spec_pad(out_magnitude, trc_magnitude)
        out_phase = self.spec_pad(out_phase, trc_phase)

        return out_magnitude,trc_magnitude, out_phase,trc_phase
    
    def run(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
        if mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.unsqueeze(0)
        if mixed_waveforms.ndim == 1:
            mixed_waveforms = mixed_waveforms.unsqueeze(0).unsqueeze(0)
        
        
        mixed_waveforms = mixed_waveforms[..., :self.audio_len].to(self.device)
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        magnitude, phase = self.model(stft_mixed)
        spech_cleaned = (magnitude*torch.exp(1j*phase)).detach().cpu()
        cleaned = librosa.istft(spech_cleaned.numpy(), n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center)
        return cleaned
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        out_magnitude,trc_magnitude, out_phase,trc_phase = self.forward(mixed_waveforms, speech_waveforms)
        
        loss = self.mse_lfn(out_magnitude, trc_magnitude) + self.sc_lfn(out_magnitude, trc_magnitude) + \
              self.phase_lfn(out_phase,trc_phase)
        
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
    


class SpectrogramLightningModelComplexUnet_V2(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelComplexUnet_v2(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.cmse = ComponentMSE()
        self.sc_lfn = SpectralConvergengeLoss()
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
        early_stop_callback = EarlyStopping(
                            monitor='valid_loss',
                            min_delta=0.00,
                            patience=20,
                            verbose=True,
                            mode='min'
                        )

        checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{valid_loss:.2f}',
            save_top_k=3,
            verbose=True,
            monitor='valid_loss',
            mode='min',
            save_last=True,  
            every_n_epochs=1,  
            auto_insert_metric_name=True)
        
        return [early_stop_callback, checkpoint_callback]

    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    
    @staticmethod
    def spec_pad(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) or spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
        
    def forward(self, mixed_waveforms, speech_waveforms):

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        
        trc_magnitude = torch.abs(stft_clean)
        out_real, out_imag = self.model(stft_mixed)
        out_magnitude = torch.sqrt(out_real**2 + out_imag**2)
        out_spec = torch.complex(out_real, out_imag)
        
        out_magnitude = self.spec_pad(out_magnitude, trc_magnitude)
        out_scpectrum = self.spec_pad(out_spec, stft_clean)

        return out_magnitude, trc_magnitude, out_scpectrum, stft_clean


    def run(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
        if mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.unsqueeze(0)
        if mixed_waveforms.ndim == 1:
            mixed_waveforms = mixed_waveforms.unsqueeze(0).unsqueeze(0)
        
        
        mixed_waveforms = mixed_waveforms[..., :self.audio_len].to(self.device)
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        out_real, out_imag  = self.model(stft_mixed)
        magnitude = torch.sqrt(out_real**2 + out_imag**2)
        phase = torch.arctan(out_imag / out_real)
        
        spech_cleaned = (magnitude*torch.exp(1j*phase)).detach().cpu()
        cleaned = librosa.istft(spech_cleaned.numpy(), n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center)
        return cleaned
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        out_magnitude, trc_magnitude, out_scpectrum, trc_spectrum = self.forward(mixed_waveforms, speech_waveforms)
        
        loss = self.cmse(out_scpectrum, trc_spectrum) + self.sc_lfn(out_magnitude, trc_magnitude)
        
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