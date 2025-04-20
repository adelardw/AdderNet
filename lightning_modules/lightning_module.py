import lightning as L
import torchaudio
from loaders import *
from utils import *
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.optim as optim
import torch.nn.functional as F
import librosa
import torch
import numpy as np
from torchmetrics.audio import( ScaleInvariantSignalDistortionRatio as SISDR, SignalDistortionRatio as SDR,
                                SignalNoiseRatio as SNR, ScaleInvariantSignalNoiseRatio as SISNR)


class OldSpectrogramLightningModelUnet(L.LightningModule):
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
     
    
    def get_mag_phase(self, mixed_waveforms):

        if mixed_waveforms.shape[0] == 1 and mixed_waveforms.ndim==2:
            mixed_waveforms = mixed_waveforms.repeat((2, 1))
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
        
        out, phase = self.get_mag_phase(mixed_waveforms)
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
    


class SpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        self.save_hyperparameters()
           

        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.loss_fn = nn.MSELoss()
        self.spectral_loss = SpectralConvergengeLoss()
        self.psl_loss = PhaseSensetiveLoss()
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
    def pad_or_trim(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) and spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
    
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
     
    
    def get_mag_phase(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
            
        if mixed_waveforms.ndim == 2:
            if mixed_waveforms.shape[0] > 1:
                mixed_waveforms = mixed_waveforms.sum(dim=0, keepdim=True)
            
            if mixed_waveforms.shape[-1] <= self.audio_len:
                mixed_waveforms = mixed_waveforms.unsqueeze(0)
        
        if mixed_waveforms.ndim == 3 and mixed_waveforms.shape[1] > 1:
            mixed_waveforms = mixed_waveforms.sum(dim=1, keepdim=True)
        
        if mixed_waveforms.shape[-1] > self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            num_audio_segments_in_wf = self.audio_shape // self.audio_len + 1
            need2pad =  self.audio_len * num_audio_segments_in_wf - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device)
            mixed_waveforms = torch.cat(torch.split(mixed_waveforms, 
                                                    split_size_or_sections=self.audio_len, dim=-1))[:, None, :]
  
        if mixed_waveforms.shape[-1] <= self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            need2pad = self.audio_len - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device) 
        
        mixed_waveforms = mixed_waveforms.repeat((1, 2, 1))
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        phase = torch.angle(stft_mixed).to(self.device) 
        magnitude = torch.abs(stft_mixed).to(self.device) 
        
        with torch.no_grad():
            output_magnitude = self.model(magnitude)
            mag_padded = self.pad_or_trim(output_magnitude, magnitude)
       
        return mag_padded, phase
    
    def forward(self, mixed_waveforms):
        
        out, phase = self.get_mag_phase(mixed_waveforms)
        
        cleaned_stft = (out * torch.exp(1j * phase)).detach().cpu().numpy()
        cleaned = librosa.istft(cleaned_stft, n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center,
                                length=self.audio_len)
        
        return torch.tensor(cleaned)
    

    def run(self, mixed_waveforms):

        cleaned = self.forward(mixed_waveforms).sum(dim=1, keepdim=True)
        if cleaned.shape[0] > 1:
            cleaned = cleaned.reshape(1, -1)
        
        return cleaned[...,:self.audio_len].reshape(1, 1, self.audio_len)


    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        
        if mixed_waveforms.shape[1] >= 1 and mixed_waveforms.ndim == 3:
            mixed_waveforms = mixed_waveforms.sum(1, keepdim=True)
        elif mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        if speech_waveforms.shape[1] >= 1 and speech_waveforms.ndim ==3:
            speech_waveforms = speech_waveforms.sum(1, keepdim=True)
        elif speech_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        clean_magnitude = torch.abs(stft_clean).to(self.device)
        clean_phase = torch.angle(stft_clean).to(self.device)
        
        mixed_magnitude = torch.abs(stft_mixed).to(self.device)
        mixed_phase = torch.angle(stft_mixed).to(self.device)
        
        output_magnitude = self.model(mixed_magnitude)
        output_magnitude = self.pad_or_trim(output_magnitude, stft_mixed)
                
        loss = 0.5*self.loss_fn(output_magnitude, clean_magnitude) + \
                1.0*self.spectral_loss(output_magnitude, clean_magnitude) + \
                0.3*self.psl_loss(output_magnitude, clean_magnitude, mixed_phase, clean_phase)
                
        cleaned = torch.tensor(self.forward(mixed_waveforms))
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
    


class GiGaSpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        self.save_hyperparameters()
        
        self.alpha = nn.Parameter(torch.tensor(0.9))  
        self.beta = nn.Parameter(torch.tensor(0.7))   
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.phase_model = PhaseCorrector()
        self.loss_fn = nn.MSELoss()
        self.spectral_loss = SpectralConvergengeLoss()
        self.psl_loss = PhaseSensetiveLoss()
        self.phase_loss = PhaseLoss()
        self.group_delay_loss = GroupDelayLoss()
        self.audio_len = audio_len
        self.metric = dict(snratio = SNR(),
                            sdratio = SDR(),
                            sisdratio = SISDR(),
                            sisnratio = SISNR())
        
    def configure_optimizers(self):
        
        #optimizer = optim.Adam(self.model.parameters(), lr=3.41e-4)
        
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.phase_model.parameters(), 'lr': 1e-4},  # Меньший LR для phase_model
            {'params': [self.alpha, self.beta], 'lr': 1e-3}         # Отдельный LR для коэффициентов
        ], lr=3.41e-4)


        scheduler_warm = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
        scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warm, scheduler_cos],
            milestones=[5])

        return [optimizer], [scheduler]
    
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
    def pad_or_trim(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) and spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
    
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
     
    
    def get_mag_phase(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
            
        if mixed_waveforms.ndim == 2:
            if mixed_waveforms.shape[0] > 1:
                mixed_waveforms = mixed_waveforms.sum(dim=0, keepdim=True)
            
            if mixed_waveforms.shape[-1] <= self.audio_len:
                mixed_waveforms = mixed_waveforms.unsqueeze(0)
        
        if mixed_waveforms.ndim == 3 and mixed_waveforms.shape[1] > 1:
            mixed_waveforms = mixed_waveforms.sum(dim=1, keepdim=True)
        
        if mixed_waveforms.shape[-1] > self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            num_audio_segments_in_wf = self.audio_shape // self.audio_len + 1
            need2pad =  self.audio_len * num_audio_segments_in_wf - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device)
            mixed_waveforms = torch.cat(torch.split(mixed_waveforms, 
                                                    split_size_or_sections=self.audio_len, dim=-1))[:, None, :]
  
        if mixed_waveforms.shape[-1] <= self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            need2pad = self.audio_len - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device) 
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        phase = torch.angle(stft_mixed).to(self.device) 
        magnitude = torch.abs(stft_mixed).to(self.device)
        
        with torch.no_grad():
            output_magnitude = self.model(magnitude)
            mag_padded = self.pad_or_trim(output_magnitude, magnitude)
            output_phase =  self.phase_model(mag_padded, phase)
            phase_padded = self.pad_or_trim(output_phase, phase)
       
        return mag_padded, phase_padded
    
    def forward(self, mixed_waveforms):
        
        out, phase = self.get_mag_phase(mixed_waveforms)
        
        cleaned_stft = (out * torch.exp(1j * phase)).detach().cpu().numpy()
        cleaned = librosa.istft(cleaned_stft, n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center,
                                length=self.audio_len)
        
        return torch.tensor(cleaned)
    
    def run(self, mixed_waveforms):

        cleaned = self.forward(mixed_waveforms)
        if cleaned.shape[0] > 1:
            cleaned = cleaned.reshape(1, -1)
        
        return cleaned[...,:self.audio_len].reshape(1, 1, self.audio_len)
    

    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        
        if mixed_waveforms.shape[1] >= 1 and mixed_waveforms.ndim == 3:
            mixed_waveforms = mixed_waveforms.sum(1, keepdim=True)
        elif mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        if speech_waveforms.shape[1] >= 1 and speech_waveforms.ndim ==3:
            speech_waveforms = speech_waveforms.sum(1, keepdim=True)
        elif speech_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        clean_magnitude = torch.abs(stft_clean).to(self.device)
        clean_phase = torch.angle(stft_clean).to(self.device)
        
        mixed_magnitude = torch.abs(stft_mixed).to(self.device)
        mixed_phase = torch.angle(stft_mixed).to(self.device)
        
        
        output_magnitude = self.model(mixed_magnitude)
        output_magnitude = self.pad_or_trim(output_magnitude, stft_mixed)
        output_phase =  self.phase_model(output_magnitude, mixed_phase)
        
        psl_loss = torch.sigmoid(self.alpha)*self.psl_loss(output_magnitude, clean_magnitude, mixed_phase, clean_phase) + \
                (1 - torch.sigmoid(self.alpha))*self.psl_loss(output_magnitude, clean_magnitude, output_phase, clean_phase)
                
        phase_loss = torch.sigmoid(self.beta) * self.phase_loss(clean_phase, output_phase) + \
                (1 - torch.sigmoid(self.beta)) * self.group_delay_loss(output_phase, clean_phase)
                
        loss = 0.5*self.loss_fn(output_magnitude, clean_magnitude) + \
                1.0*self.spectral_loss(output_magnitude, clean_magnitude) + \
                phase_loss + psl_loss
                
        
        cleaned = self.forward(mixed_waveforms)
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
    


class UltraSpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        self.save_hyperparameters()
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  
        self.beta = nn.Parameter(torch.tensor(0.7))   
        self.gamma = nn.Parameter(torch.tensor(0.3))

        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        
        self.phase_model = PhaseCorrectorDilation()
        #self.phase_model = CompatiblePhaseCorrector()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralConvergengeLoss()
        self.psl_loss = PhaseSensetiveLoss()
        self.phase_loss = PhaseLoss()
        self.group_delay_loss = GroupDelayLoss()
        self.stft_loss = STFTLoss()
        #self.multi_res_loss = MultiResolutionLoss()
        self.audio_len = audio_len
        self.metric = dict(snratio = SNR(),
                            sdratio = SDR(),
                            sisdratio = SISDR(),
                            sisnratio = SISNR())
        
    def configure_optimizers(self):
        
        #optimizer = optim.Adam(self.model.parameters(), lr=3.41e-4)
        
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.phase_model.parameters(), 'lr': 1e-4},  # Меньший LR для phase_model
            {'params': [self.alpha, self.beta, self.gamma], 'lr': 1e-3}         # Отдельный LR для коэффициентов
        ], lr=3.41e-4)


        scheduler_warm = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
        scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warm, scheduler_cos],
            milestones=[5])

        return [optimizer], [scheduler]
    
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
            filename="best-checkpoint-{epoch:02d}-{valid_snr:.2f}",
            save_last=True,
            every_n_epochs=1
        )
        
        return [checkpoint_callback_1, checkpoint_callback_2]
    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    
    @staticmethod
    def pad_or_trim(model_output, spectrogram):

        if spectrogram.size(2) >= model_output.size(2) and spectrogram.size(3) >= model_output.size(3):
            padding_rows = spectrogram.size(2) - model_output.size(2)  
            padding_cols = spectrogram.size(3) - model_output.size(3)
            out = F.pad(model_output, (0, padding_cols, 0, padding_rows))  

        else:
            out = model_output[:, :, :spectrogram.shape[-2], :spectrogram.shape[-1]]
             
        return out
    
    
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
     
    
    def get_mag_phase(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
            
        if mixed_waveforms.ndim == 2:
            if mixed_waveforms.shape[0] > 1:
                mixed_waveforms = mixed_waveforms.sum(dim=0, keepdim=True)
            
            if mixed_waveforms.shape[-1] <= self.audio_len:
                mixed_waveforms = mixed_waveforms.unsqueeze(0)
        
        if mixed_waveforms.ndim == 3 and mixed_waveforms.shape[1] > 1:
            mixed_waveforms = mixed_waveforms.sum(dim=1, keepdim=True)
           

        if mixed_waveforms.shape[-1] > self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            num_audio_segments_in_wf = self.audio_shape // self.audio_len + 1
            need2pad =  self.audio_len * num_audio_segments_in_wf - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device)
            mixed_waveforms = torch.cat(torch.split(mixed_waveforms, 
                                                    split_size_or_sections=self.audio_len, dim=-1))[:, None, :]
  
        if mixed_waveforms.shape[-1] <= self.audio_len:
            self.audio_shape = mixed_waveforms.shape[-1]
            need2pad = self.audio_len - self.audio_shape
            mixed_waveforms = F.pad(mixed_waveforms,(0, need2pad)).to(self.device) 
  
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        phase = torch.angle(stft_mixed).to(self.device) 
        magnitude = torch.abs(stft_mixed).to(self.device)
        
        with torch.no_grad():
            output_magnitude = self.model(magnitude)
            mag_padded = self.pad_or_trim(output_magnitude, magnitude)
            output_phase =  self.phase_model(mag_padded, phase)
            phase_padded = self.pad_or_trim(output_phase, phase)

        return mag_padded, phase_padded
    
    def forward(self, mixed_waveforms):
        
        out, phase = self.get_mag_phase(mixed_waveforms)
        
        cleaned_stft = (out * torch.exp(1j * phase)).detach().cpu().numpy()
        cleaned = librosa.istft(cleaned_stft, n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center,
                                length=self.audio_len)
        
        return torch.tensor(cleaned)
    
    def run(self, mixed_waveforms):

        cleaned = self.forward(mixed_waveforms)
        if cleaned.shape[0] > 1:
            cleaned = cleaned.reshape(1, -1)

        return cleaned[...,:self.audio_shape].reshape(1, 1, self.audio_shape)

    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        
        if mixed_waveforms.shape[1] >= 1 and mixed_waveforms.ndim == 3:
            mixed_waveforms = mixed_waveforms.sum(1, keepdim=True)
        elif mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        if speech_waveforms.shape[1] >= 1 and speech_waveforms.ndim ==3:
            speech_waveforms = speech_waveforms.sum(1, keepdim=True)
        elif speech_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.sum(0, keepdim=True)

        stft_clean = self.stft(speech_waveforms).to(self.device)
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        
        clean_magnitude = torch.abs(stft_clean).to(self.device)
        clean_phase = torch.angle(stft_clean).to(self.device)
        
        mixed_magnitude = torch.abs(stft_mixed).to(self.device)
        mixed_phase = torch.angle(stft_mixed).to(self.device)
        cleaned = self.forward(mixed_waveforms)
        
        output_magnitude = self.model(mixed_magnitude)
        output_magnitude = self.pad_or_trim(output_magnitude, stft_mixed)
        output_phase =  self.phase_model(output_magnitude, mixed_phase)
        stft_pred = output_magnitude * torch.exp(1j * output_phase)

        psl_loss = torch.sigmoid(self.alpha)*self.psl_loss(output_magnitude, clean_magnitude, mixed_phase, clean_phase) + \
                (1 - torch.sigmoid(self.alpha))*self.psl_loss(output_magnitude, clean_magnitude, output_phase, clean_phase)
                
        phase_loss = torch.sigmoid(self.beta) * self.phase_loss(clean_phase, output_phase) + \
                (1 - torch.sigmoid(self.beta)) * self.group_delay_loss(output_phase, clean_phase)

        reconstruct_loss = torch.sigmoid(self.gamma)*self.l2_loss(output_magnitude, clean_magnitude) + \
                            (1 - torch.sigmoid(self.gamma))*self.l1_loss(output_magnitude, clean_magnitude)
                            
        stft_loss = self.stft_loss(stft_clean, stft_pred)
        spectral_loss = self.spectral_loss(output_magnitude, clean_magnitude)        
        multi_res_loss = self.multi_res_loss(speech_waveforms.to(self.device), cleaned.to(self.device))
        

        loss = reconstruct_loss  + phase_loss + psl_loss + spectral_loss + 0.1*stft_loss + 0.1*multi_res_loss
                
        
        self.compute_metrics(cleaned.to('cpu'), speech_waveforms)
    
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