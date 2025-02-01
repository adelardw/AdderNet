from lightning_module import *
from loaders import *
from pipeline import *
import yaml

cfg_path = 'configs/denoise_model_v1_cfg.yaml'
train_loader, val_loader, test_loader = get_loaders(speech_dirs=["dev-clean.tar", "test-clean.tar"],
                                                    noise_dir="./wham_noise//wham_noise",
                                                    batch_size=64,
                                                    padding_strategy=None)




model = SpectrogramLightningModelUnet(**load_cfg(cfg_path))
trainer = L.Trainer(accelerator="auto",max_epochs=300,logger=True)                  
trainer.fit(model, train_loader, val_loader)