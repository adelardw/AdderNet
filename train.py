from loaders import *
from pipeline import *
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.optim as optim
import lightning as L
import torchaudio

train_loader, val_loader, test_loader = get_loaders(speech_dirs=["dev-clean", "test-clean"],
                                                    noise_dir="./wham_noise//wham_noise",
                                                    batch_size=8,
                                                    padding_strategy=None)




encoder_attributes= dict(in_channels=2,
                        out_channels = [8, 16 , 32], 
                        kernel_sizes = [3, 5, 5],
                        use_mobile = False,
                        act_func = nn.Tanh(),
                        do_bn = [False, True, True],
                        do_sc = [False, True, True],
                        dp = [0.2, 0.2],
                        num_blocks = 3)




decoder_attributes = dict(in_channels=32, 
                        out_channels = [16, 8, 1], 
                        kernel_sizes = [5, 5, 3],
                        use_mobile = False,
                        act_func =nn.Tanh(),
                        do_bn = [False, True, True],
                        do_sc = [True, True, True], 
                        dp = [0.3, 0.4, 0.4],
                        num_blocks = 3)


model_attributes = dict(n_fft =1022,
                    hop_length = 250,
                    center = True,
                    input_signal_size = 80000,
                    encoder_parameters = encoder_attributes,
                    decoder_parameters = decoder_attributes)

class SpectrogramLightningModel(L.LightningModule):
    def __init__(self, attributes):
        super().__init__()
        
        self.n_fft = attributes['n_fft'] 
        self.hop_length = attributes['hop_length']
        self.stft = torchaudio.transforms.Spectrogram(n_fft = self.n_fft, hop_length=self.hop_length, center=False, normalized=True)
        self.model = DenoisingModel(**attributes)
        
        self.mse = nn.MSELoss()
        self.sisdr = SiSDRLoss() 
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.model.parameters(), lr=1.41e-4)
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
    def pad(outs, inputs):

        if outs.size(2) >= inputs.size(2) and outs.size(3) >= inputs.size(3):
            padding_rows = outs.size(2) - inputs.size(2)  
            padding_cols = outs.size(3) - inputs.size(3)
            padded = F.pad(inputs, (0, padding_cols, 0, padding_rows))  

        elif outs.size(2) <= inputs.size(2) and outs.size(3) <= inputs.size(3):
            padding_rows =  - (outs.size(2) - inputs.size(2) )
            padding_cols = -(outs.size(3) - inputs.size(3))
            padded = F.pad(outs, (0, padding_cols, 0, padding_rows))  

        return padded
    

    def forward(self, mixed_waveforms, speech_waveforms):
        

        stft_mixed = self.stft(mixed_waveforms).to(self.device)
        stft_clean = self.stft(speech_waveforms).to(self.device)

        output = self.model(stft_mixed)
        if output.shape != stft_clean.shape:
            output = self.pad(output, stft_clean)

        return output, stft_clean
    
    def _step(self, batch, kind):

        mixed_waveforms, speech_waveforms = batch
        output, stft_clean = self.forward( mixed_waveforms, speech_waveforms)
        
        loss = self.mse(output, stft_clean)
        #metric = self.sisdr(cleaned, speech_waveforms)
        
        metrics = {
            f"{kind}_metric": loss,
            f"{kind}_loss": loss,
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True)

        return loss
    

model = SpectrogramLightningModel(model_attributes)
trainer = L.Trainer(accelerator="auto",max_epochs=100,logger=True)                  
                            
                            

trainer.fit(model, train_loader, val_loader)