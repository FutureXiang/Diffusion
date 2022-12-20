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

## Results & Observations
### Discrete-time Markov chain: DDPM/DDIM
Training / sampling in **[DDPM](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) / [DDIM](https://arxiv.org/pdf/2010.02502.pdf)** style (with DDPM 35.7M basic network):
| model                                                                | EMA | DDPM | DDIM(100 steps) |
|----------------------------------------------------------------------|-----|------|-----------------|
| (reported in papers)                                                 | yes | 3.17 | 4.16            |
| ([official checkpoint](https://github.com/pesser/pytorch_diffusion)) | yes | 3.13 | 4.10            |
| unconditional                                                        | yes | 3.00 | 3.59            |
| conditional ([CFG](https://arxiv.org/pdf/2207.12598.pdf), w=0.3)     | yes | 3.19 | 3.39            |

Some observations:
- BigGAN-style up/downsampling (proposed by [Score-based SDE](https://arxiv.org/pdf/2011.13456.pdf) and [ADM](https://proceedings.neurips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)) doesn't seem to work on discrete-time models.
- Ablated attention (resolution @ 32,16,8; heads=4 & dim=64, proposed by ADM) has little FID improvement on CIFAR-10, but costs heavily on memory & speed.

### Continuous-time SDE: EDM
Unconditional training / sampling in **[EDM](https://arxiv.org/pdf/2206.00364.pdf)** style (with DDPM 35.7M basic network):
|eta/steps| steps=18 | steps=50 | steps=100 |
|---------|----------|----------|-----------|
| eta=0.0 |   3.39   |   3.64   |    3.68   |
| eta=0.5 |   3.10   |   2.95   |    2.93   |
| eta=1.0 |   3.12   |   2.84   |    2.97   |

With BigGAN-style up/downsampling blocks (`use_res_for_updown=True`), FID further improves to:
|eta/steps| steps=18 | steps=50 | steps=100 |
|---------|----------|----------|-----------|
| eta=0.0 |   3.01   |   3.08   |    3.10   |
| eta=0.5 |   3.10   |   2.73   |    2.57   |
| eta=1.0 |   3.40   |   2.72   |  **2.50** |

Note that:
- eta $\eta = \frac{S_{churn} / N}{\sqrt{2}-1}$ controls stochasticity. `eta=0.0` is equivalent to a deterministic sampler.
- EDM uses a 2nd order sampler and the actual neural function evaluations (NFEs) equal to $2\times$steps.

## Citations
This implementation is based on / inspired by:
- DDPM, 35.7M unconditional U-Net: https://github.com/hojonathanho/diffusion (official DDPM TensorFlow repo)
- 35.7M unconditional U-Net: https://github.com/pesser/pytorch_diffusion (exact translation from TensorFlow to PyTorch)
- Ablated conditional U-Net: https://github.com/openai/guided-diffusion (official Classifier Guidance repo)
- DDIM sampling: https://github.com/ermongroup/ddim (official DDIM repo)
- EDM: https://github.com/NVlabs/edm (official EDM repo)
- Classifier-free Guidance sampling: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
- Others: https://github.com/lucidrains/denoising-diffusion-pytorch (a nonequivalent PyTorch implementation of DDPM)