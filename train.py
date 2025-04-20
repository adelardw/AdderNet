from lightning_modules.lightning_module import *
from loaders import *
from utils import *

cfg_path = 'configs/denoise_model_v1_cfg.yaml'
ckpt_path = '/home/ys/diploma/lightning_logs/version_30/checkpoints/best-checkpoint-epoch=00-valid_loss=0.79.ckpt'
train_loader, val_loader, test_loader = get_loaders(speech_dirs=["dataset/dev-clean", "dataset/test-clean.tar"],
                                                    noise_dir="dataset/wham_noise//wham_noise",
                                                    batch_size=3,
                                                    padding_strategy=None)




#giga_model = GiGaSpectrogramLightningModelUnet.load_from_checkpoint(ckpt_path,**load_cfg(cfg_path))   #(**load_cfg(cfg_path))

model = UltraSpectrogramLightningModelUnet(**load_cfg(cfg_path)) #.load_from_checkpoint(ckpt_path,**load_cfg(cfg_path))


#model.model.load_state_dict(giga_model.model.state_dict())
#del giga_model 

trainer = L.Trainer(accelerator="auto",max_epochs=300,logger=True)                  
trainer.fit(model, train_loader, val_loader)