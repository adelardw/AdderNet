import lightning as L
import torchaudio
from loaders import *
from pipeline import *
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.optim as optim
import torch.nn.functional as F
import librosa


encoder_attributes= dict(in_channels=2,
                        out_channels = [8, 32, 32, 64, 64, 128, 128, 256], 
                        kernel_sizes = [9, 7, 7, 5, 5, 3, 3, 3],
                        use_mobile = True,
                        act_func = [nn.ELU()] * 6 + [nn.Tanh()] * 2 ,
                        do_bn = [True],
                        do_sc = [True],
                        dp = [0.2, 0.2, 0.4],
                        num_blocks = 8)




decoder_attributes = dict(in_channels=256, 
                        out_channels = [128, 128, 64, 64, 32, 32, 8, 1], 
                        kernel_sizes = [3, 3, 3, 5, 5, 7, 7, 9],
                        use_mobile = False,
                        act_func =nn.ELU(),
                        do_bn = [True],
                        do_sc = [True], 
                        dp = [0.3, 0.4, 0.4],
                        num_blocks = 8)

stft_attributes = dict(n_fft = 2046, 
                    hop_length=123,
                    win_length=256,
                    window_fn=torch.hann_window, 
                    center=False,
                    normalized=False, power = 1)

model_attributes = dict(
                    encoder_parameters = encoder_attributes,
                    decoder_parameters = decoder_attributes)

audio_len = 128000 # sample rate = 16k


class SpectrogramLightningModelUnet(L.LightningModule):
    def __init__(self, model_attributes, stft_attributes, audio_len = audio_len):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(**stft_attributes)
        self.model = DenoisingModelUnet(**model_attributes)
        self.loss_fn = nn.MSELoss()
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
        
        
        mixed_waveforms = mixed_waveforms[..., :self.audio_len].to(self.device)
        
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