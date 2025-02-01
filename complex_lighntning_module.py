from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
import torch.optim as optim
import torch.nn.functional as F
import librosa
import lightning as L
import torchaudio
from denoising.loaders import *
from denoising.pipeline import *




class SpectrogramLightningModelComplexUnet(L.LightningModule):
    def __init__(self, encoder_parameters,decoder_parameters, stft_attributes, audio_len = 128000):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelComplexUnet(encoder_parameters=encoder_parameters,
                                               decoder_parameters=decoder_parameters)
        self.loss_fn = CMSE()
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
        
        
        output = self.model(stft_mixed)
        
        if output.shape != stft_clean.shape:
            output = self.spec_pad(output, stft_clean)

        return output, stft_clean
    
    def run(self, mixed_waveforms):
        
        if isinstance(mixed_waveforms, np.ndarray):
            mixed_waveforms = torch.tensor(mixed_waveforms)
        if mixed_waveforms.ndim == 2:
            mixed_waveforms = mixed_waveforms.unsqueeze(0)
        if mixed_waveforms.ndim == 1:
            mixed_waveforms = mixed_waveforms.unsqueeze(0).unsqueeze(0)
        
        
        mixed_waveforms = mixed_waveforms[..., :self.audio_len]
        
        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        spec_cleaned = self.model(stft_mixed)
        cleaned = librosa.istft(spec_cleaned.detach().cpu().numpy(), n_fft=self.stft.n_fft,hop_length=self.stft.hop_length,
                                win_length=self.stft.win_length, center=self.stft.center)
        return cleaned
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        output, stft_clean = self.forward(mixed_waveforms, speech_waveforms)
        
        loss = self.loss_fn(output, stft_clean)
        
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