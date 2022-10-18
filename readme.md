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
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 cifar.py       --config config/cifar_unconditional.yaml`
#### Conditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 cifar_guide.py --config config/cifar_conditional.yaml`

### Sampling
#### Unconditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 sample.py       --config config/cifar_unconditional.yaml --ema`
#### Conditional
`CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 sample_guide.py --config config/cifar_conditional.yaml   --w 0.3`

### Compute FID
#### Preprocessing
Use `extract_cifar10_pngs.ipynb` to convert CIFAR-10 training dataset to 50000 pngs.
#### Compute FID score
`python -m pytorch_fid data/cifar10-pngs/ xxxxxx/generated_ep1999_w0.3_ddim_steps100_eta0.0/pngs/ --device cuda:2`

## Results & Observations
### Sampling from [official checkpoint weights](https://github.com/pesser/pytorch_diffusion)
Sample 50000 images by adapted `sample.py` and `DDPM.py`, with official checkpoints:
- DDPM, with EMA, FID = 3.13
- DDIM, with EMA, FID = 4.10
    - The $\bar\alpha_t$ schedule proposed by [DDIM](https://arxiv.org/pdf/2010.02502.pdf) (i.e. $\bar\alpha_0 := 1$, implemented by `torch.cat([torch.tensor([0.0]), torch.linspace(beta1, beta2, T)])`) performs worse on FID than the bugged/unofficial one (implemented by `beta_t = torch.linspace(beta1, beta2, T + 1)`) ?
### Training from scratch
- [TODO]

### Some observations
- BigGAN up/downsampling (proposed by [DDPM++](https://openreview.net/pdf?id=PxTIG12RRHS) and [Beat GANs](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)) doesn't seem to work on conditional model.
- EMA doesn't seem to work on conditional model.
- Ablated attention (resolution @ 32,16,8; heads=4 & dim=64, proposed by [Beat GANs](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)) has little FID improvement on CIFAR-10, but costs heavily on memory & speed.

## Citations
This implementation is based on / inspired by:
- DDPM, 35.7M unconditional U-Net: https://github.com/hojonathanho/diffusion (official DDPM TensorFlow repo)
- 35.7M unconditional U-Net: https://github.com/pesser/pytorch_diffusion (exact translation from TensorFlow to PyTorch)
- Ablated conditional U-Net: https://github.com/openai/guided-diffusion (official Classifier Guidance repo)
- DDIM sampling: https://github.com/ermongroup/ddim (official DDIM repo)
- Classifier-free Guidance sampling: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
- Others: https://github.com/lucidrains/denoising-diffusion-pytorch (a nonequivalent PyTorch implementation of DDPM)