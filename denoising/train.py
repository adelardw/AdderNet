from lightning_module import *
from loaders import *
from pipeline import *

train_loader, val_loader, test_loader = get_loaders(speech_dirs=["dev-clean.tar", "test-clean.tar"],
                                                    noise_dir="./wham_noise//wham_noise",
                                                    batch_size=64,
                                                    padding_strategy=None)


model = SpectrogramLightningModelUnet(model_attributes=model_attributes,
                           stft_attributes=stft_attributes)
trainer = L.Trainer(accelerator="auto",max_epochs=1000,logger=True)                  
                                        

trainer.fit(model, train_loader, val_loader)