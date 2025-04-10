from lightning_modules.lightning_module import *
from utils import cfg_loader

ckpt_path = 'configs/last.ckpt'
cfg_path = 'configs/denoise_model_v1_cfg.yaml'
save_cleaned_audio_path = 'denoised_example.wav'

model = SpectrogramLightningModelUnet.load_from_checkpoint(ckpt_path, **cfg_loader.load_cfg(cfg_path))
audio, rate = torchaudio.load('example.wav')
denoised = model.run(audio)[0]
torchaudio.save(save_cleaned_audio_path, torch.tensor(denoised), rate)