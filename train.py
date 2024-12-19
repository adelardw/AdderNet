from loaders import *
from pipeline import *
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.optim as optim
import lightning as L
import librosa

train_loader, val_loader, test_loader = get_loaders(speech_dirs=["dev-clean", "test-clean"],
                                                    noise_dir="./wham_noise//wham_noise",
                                                    batch_size=8,
                                                    padding_strategy=None)




class LightningModel(L.LightningModule):
    def __init__(self, attributes):
        super().__init__()
        
        self.n_fft = attributes['n_fft'] 
        self.hop_length = attributes['hop_length']
        self.model = DenoisingModel(**attributes)
        #self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sisdr = SiSDRLoss() 
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        """linear = optim.lr_scheduler.LinearLR(optimizer,
                                            start_factor=1e-5,
                                            end_factor=1e-4,
                                            total_iters=4)"""
        
        
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10, eta_min=1e-10)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        #scheduler =  optim.lr_scheduler.SequentialLR(optimizer, schedulers = [linear, decay], milestones=[2])
        

        lr_scheduler_config = {
            "scheduler": scheduler,

            "interval": "epoch",

            "frequency": 1,
        }

        return [optimizer] # [lr_scheduler_config]
    
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
    def masking(outs, inputs):

        if outs.size(2) >= inputs.size(2) and outs.size(3) >= inputs.size(3):
            padding_rows = outs.size(2) - inputs.size(2)  
            padding_cols = outs.size(3) - inputs.size(3)
            padded = F.pad(inputs, (0, padding_cols, 0, padding_rows))  
            masked = outs * masked

        elif outs.size(2) < inputs.size(2) and outs.size(3) < inputs.size(3):
            padding_rows =  - (outs.size(2) - inputs.size(2) )
            padding_cols = -(outs.size(3) - inputs.size(3))
            padded = F.pad(outs, (0, padding_cols, 0, padding_rows))  
            masked = inputs * padded

        return masked
    
    def _step(self, batch, kind):
        mixed_waveforms, speech_waveforms = batch

        stft_mixed = librosa.stft(mixed_waveforms.cpu().detach().numpy(), n_fft = self.n_fft, hop_length= self.hop_length)
        x = torch.tensor(np.abs(stft_mixed), requires_grad=True).to(self.device)
        mask = self.model(x)
        masked = self.masking(mask, x)
        reverse_clear_mixed = librosa.istft(masked.cpu().detach().numpy(), n_fft=self.n_fft, hop_length=self.hop_length)
        cleaned = torch.tensor(reverse_clear_mixed, requires_grad=True).to(self.device)
        cleaned = cleaned[:, 0, None, :]
        
        loss = self.mse(cleaned, speech_waveforms)
        metric = self.sisdr(cleaned, speech_waveforms)
        
        metrics = {
            f"{kind}_metric": metric,
            f"{kind}_loss": loss,
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True)

        return loss
    

