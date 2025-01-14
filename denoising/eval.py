from lightning_module import *

model = SpectrogramLightningModelUnet.load_from_checkpoint('ep_130.ckpt',
                                                            model_attributes=model_attributes,
                                                            stft_attributes=stft_attributes)
audio, rate = torchaudio.load('example.wav')
denoised = model.run(audio)[0]