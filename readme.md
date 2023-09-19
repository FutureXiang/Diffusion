# Minimal Diffusion
This is a multi-gpu PyTorch implementation of Diffusion Models with Classifier-Free Guidance (CFG).

This repo contains:
- Training, sampling and FID evaluation code for diffusion models, including
  - Frameworks:
    - `DDPM.py`: Unconditional DDPM & DDIM
    - `DDPM_guide.py`: Class-conditional DDPM & DDIM with CFG sampling
    - `EDM.py`:  Unconditional EDM
    - `EDM_guide.py`: Class-conditional EDM with CFG sampling
  - Networks:
    - `unet.py` & `unet_guide.py`: Unconditional & conditional U-Net (DDPM & DDPM++)
    - `uvit.py`: Unconditional & conditional Vision Transformer (U-ViT)
  - Datasets:
    - CIFAR-10


## Requirements
- In addition to PyTorch environments, please install:
  ```sh
  conda install pyyaml
  pip install pytorch-fid ema-pytorch timm einops
  ```
- At least 4 3080ti GPUs are recommended to train diffusion models on CIFAR-10. With automatic mixed precision enabled and 4 GPUs, training a basic 35.7M UNet on CIFAR-10 takes ~14 hours.
- The `pytorch-fid` requires image files to calculate the FID metric. Please refer to `extract_cifar10_pngs.ipynb` to unpack the CIFAR-10 training dataset into 50000 `.png` image files.


## Usage

### Training
To train a diffusion model with 4 GPUs and AMP enabled, for example, run:
```sh
python -m torch.distributed.launch --nproc_per_node=4
  # unconditional
  cifar.py       --config config/cifar_unconditional.yaml --use_amp

  # conditional
  cifar_guide.py --config config/cifar_conditional.yaml   --use_amp
```

### Sampling
To generate 50000 image samples with a trained diffusion model, for example, run:
```sh
python -m torch.distributed.launch --nproc_per_node=4
  # unconditional
  # deterministic fast sampling (i.e. DDIM 100 steps / EDM 18 steps)
  sample.py       --config config/cifar_unconditional.yaml --use_amp
  # stochastic sampling (i.e. DDPM 1000 steps)
  sample.py       --config config/cifar_unconditional.yaml --use_amp --mode DDPM

  # conditional
  # deterministic fast sampling (i.e. DDIM 100 steps / EDM 18 steps)
  sample_guide.py --config config/cifar_conditional.yaml   --use_amp
  # stochastic sampling (i.e. DDPM 1000 steps)
  sample_guide.py --config config/cifar_conditional.yaml   --use_amp --mode DDPM
```

### Compute FID
To calculate the FID metric on the training set, for example, run:
```sh
python -m pytorch_fid   data/cifar10-pngs/  2000_unconditional/EMAgenerated_ep1999_ddpm/pngs/
```

## Results
|             Config             |      Model      |   Network   | Cond | FID (best sampler/NFE) | FID (fast sampler/NFE) |
|--------------------------------|-----------------|-------------|------|------------------------|------------------------|
| `cifar_unconditional.yaml`     |     DDPM        |  35.7M UNet | no   | 3.03 (DDPM / 1000)     | 3.54 (DDIM / 100)      |
| `cifar_unconditional_cosine.yaml`|DDPM (cosine)  |  35.7M UNet | no   | 3.18 (DDPM / 1000)     | 3.56 (DDIM / 100)      |
| `cifar_conditional.yaml`       |     DDPM        |  38.6M UNet | yes  | 3.27 (DDPM / 2000)     | 3.46 (DDIM / 200)      |
| `cifar_unconditional_vit.yaml` |     DDPM        |  44.3M UViT | no   | 3.33 (DDPM / 1000)     | 4.51 (DDIM / 100)      |
| `cifar_conditional_vit.yaml`   |     DDPM        |  44.3M UViT | yes  | 2.87 (DDPM / 2000)     | 3.19 (DDIM / 200)      |
| `cifar_unconditional_EDM.yaml` |     EDM         |  56.5M UNet | no   | 2.22 (EDM  / 35)       | 2.22 (EDM / 35)        |
| `cifar_conditional_EDM.yaml`   |     EDM         |  61.1M UNet | yes  | 2.06 (EDM  / 53)       | 2.06 (EDM / 53)        |
|            N/A                 |     EDM         |  44.3M UViT | no   |      NOT WORKING       |      NOT WORKING       |


## Acknowledgments
This repository is built on numerous open-source codebases such as [DDPM](https://github.com/hojonathanho/diffusion), [DDPM-pytorch](https://github.com/pesser/pytorch_diffusion), [DDIM](https://github.com/ermongroup/ddim), [EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), [U-ViT](https://github.com/baofff/U-ViT), and [CFG-MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST).
