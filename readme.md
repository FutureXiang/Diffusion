# Minimal Classifier-free-DDIM
Minimal implementation of Denoising Diffusion Probabilistic Models (DDPM) with Classifier-free Guidance and DDIM fast sampling.

## Additional Dependencies
```
pip install pytorch-fid
pip install ema-pytorch
```

## Scripts

### Training
#### Unconditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 cifar.py       --config config/cifar_unconditional.yaml --use_amp`
#### Conditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 cifar_guide.py --config config/cifar_conditional.yaml   --use_amp`

### Sampling
#### Unconditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 sample.py       --config config/cifar_unconditional.yaml --use_amp --ema`
#### Conditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 sample_guide.py --config config/cifar_conditional.yaml   --use_amp --ema --w 0.3`

### Compute FID
#### Preprocessing
Use `extract_cifar10_pngs.ipynb` to convert CIFAR-10 training dataset to 50000 pngs.
#### Compute FID score
`python -m pytorch_fid data/cifar10-pngs/ xxxxxx/generated_ep1999_w0.3_ddim_steps100_eta0.0/pngs/ --device cuda:2`

## Results
|             Config             | Diffusion Model |   Network   | Conditional |      FID (best)        |  FID (fast sampling)   |           Note          |
|--------------------------------|-----------------|-------------|-------------|------------------------|------------------------|-------------------------|
| `cifar_unconditional.yaml`     |     DDPM        |  35.7M UNet |     no      | 3.11 (DDPM, 1000 NFE)  | 3.60 (DDIM, 100 NFE)   | official: 3.17/4.16     |
| `cifar_unconditional_cosine.yaml`|DDPM (cosine)  |  35.7M UNet |     no      | 3.13 (DDPM, 1000 NFE)  | 3.53 (DDIM, 100 NFE)   | 700 epochs training, more efficient |
| `cifar_conditional.yaml`       |     DDPM        |  38.6M UNet |     yes     | 3.19 (DDPM, 2000 NFE)  | 3.39 (DDIM, 200 NFE)   | guidance weight `w=0.3` |
| `cifar_unconditional_vit.yaml` |     DDPM        |  44.3M UViT |     no      | 3.18 (DDPM, 1000 NFE)  | 4.15 (DDIM, 100 NFE)   | official: 3.11          |
| `cifar_conditional_vit.yaml`   |     DDPM        |  44.3M UViT |     yes     | 2.82 (DDPM, 2000 NFE)  | 3.32 (DDIM, 200 NFE)   | guidance weight `w=0.3` |
| `cifar_unconditional_EDM.yaml` |     EDM         |  56.5M UNet |     no      | 2.19 (EDM ODE, 35 NFE) | 2.19 (EDM ODE, 35 NFE) | official: 2.05          |
| `cifar_conditional_EDM.yaml`   |     EDM         |  61.1M UNet |     yes     | 2.00 (EDM ODE, 53 NFE) | 2.00 (EDM ODE, 53 NFE) | guidance weight `w=0.3` |
|            N/A                 |     EDM         |  44.3M UViT |     no      |      NOT WORKING       |      NOT WORKING       | weird artifacts at the last row/column of images |


## Citations
This implementation is based on / inspired by:
- DDPM, 35.7M unconditional U-Net: https://github.com/hojonathanho/diffusion (official DDPM TensorFlow repo)
- 35.7M unconditional U-Net: https://github.com/pesser/pytorch_diffusion (exact translation from TensorFlow to PyTorch)
- Ablated conditional U-Net: https://github.com/openai/guided-diffusion (official Classifier Guidance repo)
- DDIM sampling: https://github.com/ermongroup/ddim (official DDIM repo)
- EDM: https://github.com/NVlabs/edm (official EDM repo)
- Classifier-free Guidance sampling: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
- Others: https://github.com/lucidrains/denoising-diffusion-pytorch (a nonequivalent PyTorch implementation of DDPM)