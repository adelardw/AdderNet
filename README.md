# AudioDenoisingNet

Lightweight CNN-based speech enhancement (audio denoising) in the spectrogram domain. The model takes the magnitude spectrogram of noisy audio, reconstructs a clean magnitude with a U-Net, and recovers phase with a small dedicated phase-reconstruction head. It is designed to favour **inference speed** (telephony, mobile, IoT, ASR front-ends) over raw SOTA quality, while still staying competitive on standard speech-enhancement benchmarks.

This repository contains the code accompanying the master's thesis *"Investigation of deep-neural-network-based denoising methods for audio signals"* (HSE University, Applied Mathematics and Informatics / Machine Learning and Data Analysis).

## Key ideas

- **STFT magnitude in, clean magnitude out.** Audio is transformed with a Short-Time Fourier Transform (Hann window). A U-Net encoder–decoder with skip connections cleans the magnitude spectrogram, and the signal is reconstructed with the inverse STFT.
- **Explicit phase handling.** A small (~83K-parameter) phase-reconstruction head takes the noisy phase together with the cleaned magnitude and predicts a clean phase. This removes the negative-SNR artifact you get from naively reusing the noisy phase.
- **Composite loss.** Training combines magnitude losses (L1/L2), Spectral Convergence Loss, Phase Sensitive Loss, Group Delay Loss, a plain phase loss, and an STFT loss. Loss-term weights `α`, `β` are learnable.
- **~2M parameters total.** The whole pipeline (1.9M U-Net + ~83K phase head) stays under 2M parameters.

## Results

Evaluated on **VoiceBank + DEMAND** and **LibriSpeech + WHAM!**.

| Metric | VoiceBank + DEMAND | LibriSpeech + WHAM! |
|---|---|---|
| SNR | 3.84 dB | 8.23 dB |
| SDR | 5.40 dB | 7.96 dB |
| SI-SDR | 5.20 dB | 7.65 dB |
| SI-SNR | 5.20 dB | 7.66 dB |
| STOI | 0.902 | 0.900 |
| PESQ-NB | 2.699 | 2.426 |
| PESQ-WB | 1.876 | 1.784 |

Inference speed (single audio clip, Nvidia Tesla T4, VoiceBank + DEMAND):

| Model | Params | Time |
|---|---|---|
| DCCRN | 3M+ | 0.0106 s |
| MP-SEUnet | 2.2M | 0.0226 s |
| TF-Locoformer | 7.8M | 0.0240 s |
| **AudioDenoisingNet (GPU)** | **< 2M** | **0.0047 s** |
| **AudioDenoisingNet (GPU + CPU)** | **< 2M** | **0.065 s** |

Quality is below transformer/Mamba SOTA models (e.g. TF-Locoformer reaches SI-SDR > 15 dB), but this model is the fastest on GPU in the comparison above, which is the trade-off it is built for. The STFT/ISTFT and padding/reshape steps run on CPU and dominate the CPU-side latency.

## Repository structure

```
AudioDenoisingNet/
├── configs/             # YAML model/training configs (e.g. denoise_model_v1_cfg.yaml)
├── lightning_modules/   # PyTorch Lightning modules (model + train/val/test steps)
├── loaders/             # Dataset and DataLoader construction (get_loaders)
├── utils/               # Helpers, incl. config loader (cfg_loader.load_cfg)
├── onnx_converts/       # ONNX export scripts
├── examples/            # Example audio / usage
├── ckpts/               # Checkpoints
├── train.py             # Training entrypoint
├── eval.py              # Inference / denoising entrypoint
├── test_model.ipynb     # Exploration / evaluation notebook
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/adelardw/AudioDenoisingNet.git
cd AudioDenoisingNet
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchaudio`, `librosa`, `lightning`.

## Data

Training uses **LibriSpeech** (clean speech) mixed with **WHAM!** (noise). Expected layout:

```
dataset/
├── dev-clean/                  # LibriSpeech clean speech
├── test-clean.tar              # LibriSpeech test split
└── wham_noise/wham_noise/      # WHAM! noise
```

- LibriSpeech: https://www.openslr.org/12
- WHAM!: https://wham.whisper.ai/
- VoiceBank + DEMAND (benchmark): https://datashare.ed.ac.uk/handle/10283/2791

## Usage

### Training

Point `train.py` at a config and your dataset directories, then run:

```bash
python train.py
```

Internally this looks like:

```python
from lightning_modules.lightning_module import *
from loaders import *
from utils import *
import lightning as L

cfg_path = 'configs/denoise_model_v1_cfg.yaml'

train_loader, val_loader, test_loader = get_loaders(
    speech_dirs=["dataset/dev-clean", "dataset/test-clean.tar"],
    noise_dir="dataset/wham_noise/wham_noise",
    batch_size=16,
    padding_strategy=None,
)

model = UltraSpectrogramLightningModelUnet(**load_cfg(cfg_path))

trainer = L.Trainer(accelerator="auto", max_epochs=300, logger=True)
trainer.fit(model, train_loader, val_loader)
```

> The repo contains several Lightning model variants (e.g. `SpectrogramLightningModelUnet`, `UltraSpectrogramLightningModelUnet`, `GiGaSpectrogramLightningModelUnet`). Pick the one matching your config; you can also warm-start one model from another's `state_dict`.

### Inference / denoising a file

```python
import torch, torchaudio
from lightning_modules.lightning_module import *
from utils import cfg_loader

ckpt_path = 'configs/last.ckpt'
cfg_path = 'configs/denoise_model_v1_cfg.yaml'

model = SpectrogramLightningModelUnet.load_from_checkpoint(
    ckpt_path, **cfg_loader.load_cfg(cfg_path)
)

audio, rate = torchaudio.load('example.wav')
denoised = model.run(audio)[0]
torchaudio.save('denoised_example.wav', torch.tensor(denoised), rate)
```

Or just run the provided script:

```bash
python eval.py
```

For long audio, segments equal in length to the training clips can be batched and processed in parallel, then stitched back together.

## Training configuration

Final model (phase-aware), summarized from the thesis:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| LR (U-Net 1.9M) | 0.00341 |
| LR (phase head 83K) | 0.0001 |
| Weight decay | 0.2 |
| Scheduler | Linear warmup (start factor 0.01, 5 iters) → Cosine Annealing (T_max 25, eta_min 5e-5) |
| Epochs | 50 |
| Batch size | 16 |
| Bottleneck activation | Tanh |
| Losses | L1, L2, Phase Sensitive, Spectral Convergence, Group Delay, Phase, STFT |
| Total params | ~2M |

## Limitations & roadmap

- Quality trails attention/SSM-based SOTA models; the current phase handling does not fully remove phase distortion.
- CPU inference is dominated by STFT/ISTFT and tensor reshaping.
- Planned: attention/Transformer and Mamba (SSM) blocks for higher quality, and further optimization for mobile devices. For telephony, add compression/channel-distortion augmentations during training.

## Practical applications

- Cleaning voice messages in messengers (the model is trained on real-world street noise).
- Phone-call enhancement (with appropriate telephony augmentations added at training time).
- A denoising front-end in an ASR pipeline.

## Citation

If this is useful, please cite the thesis:

> Sergaev Y. S. *Investigation of deep-neural-network-based denoising methods for audio signals.* Master's thesis, HSE University, School of Computer Science, Physics and Technology.

## License

No license file is currently present in the repository. Add one (e.g. MIT) if you intend others to reuse the code.
