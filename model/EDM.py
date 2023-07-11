import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast


# ===== Normalization (helper functions)

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


class EDM(nn.Module):
    def __init__(self, nn_model,
                 sigma_data, p_mean, p_std,
                 sigma_min, sigma_max, rho,
                 S_min, S_max, S_noise,
                 device):
        '''
            EDM proposed by "Elucidating the Design Space of Diffusion-Based Generative Models".
            Args:
                `nn_model`: A network (e.g. UNet) which performs same-shape mapping.
                `device`: The CUDA device that tensors run on.
            Training parameters:
                `sigma_data`, `p_mean`, `p_std`
            Sampling parameters:
                `sigma_min`, `sigma_max`, `rho`
                `S_min`, `S_max`, `S_noise`
        '''
        super(EDM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device

        def number_to_torch_device(value):
            return torch.tensor(value).to(device)

        self.sigma_data = number_to_torch_device(sigma_data)
        self.p_mean     = number_to_torch_device(p_mean)
        self.p_std      = number_to_torch_device(p_std)
        self.sigma_min  = number_to_torch_device(sigma_min)
        self.sigma_max  = number_to_torch_device(sigma_max)
        self.rho        = number_to_torch_device(rho)
        self.S_min      = number_to_torch_device(S_min)
        self.S_max      = number_to_torch_device(S_max)
        self.S_noise    = number_to_torch_device(S_noise)

    def D_x(self, x_noised, sigma, use_amp, get_activation=False):
        '''
            Denoising with network preconditioning.
            Args:
                `x_noised`: The perturbed image tensor.
                `sigma`: The variance (or noise level) tensor.
            Returns:
                The estimated denoised image tensor `x`.
        '''
        x_noised = x_noised.to(torch.float32)
        sigma = sigma.to(torch.float32)
        # Preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Denoising
        if get_activation:
            with autocast(enabled=use_amp):
                return self.nn_model.get_activation(c_in * x_noised, c_noise.flatten())

        with autocast(enabled=use_amp):
            F_x = self.nn_model(c_in * x_noised, c_noise.flatten())
        return c_skip * x_noised + c_out * F_x

    def forward(self, x, use_amp=False):
        '''
            Training with weighted denoising loss.
            Args:
                `x`: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The weighted MSE loss tensor.
        '''
        x = normalize_to_neg_one_to_one(x)

        # Perturbation
        rnd_normal = torch.randn((x.shape[0], 1, 1, 1)).to(self.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(x)
        x_noised = x + noise * sigma

        # Weighted Denoising loss
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss_4shape = weight * ((x - self.D_x(x_noised, sigma, use_amp)) ** 2)
        return loss_4shape.mean()

    def get_feature(self, x, t, norm, num_steps=18, use_amp=False):
        x = normalize_to_neg_one_to_one(x)

        # Time steps
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        times = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        times = (sigma_max ** (1 / rho) + times / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        times = torch.cat([times, torch.zeros_like(times[:1])]) # t_N = 0
        times = reversed(times)
        # Perturbation
        sigma = times[t]
        noise = torch.randn_like(x)
        x_noised = x + noise * sigma

        ret = {}
        feats = self.D_x(x_noised, sigma, use_amp, get_activation=True)
        for blockname in feats:
            feat = feats[blockname].float()
            if len(feat.shape) == 4:
                # unet (B, C, H, W)
                feat = feat.view(feat.shape[0], feat.shape[1], -1)
                feat = torch.mean(feat, dim=2)
            elif len(feat.shape) == 3:
                # vit (B, L, D)
                feat = feat[:, self.nn_model.extras:, :] # optional
                feat = torch.mean(feat, dim=1)
            else:
                raise NotImplementedError
            if norm:
                feat = torch.nn.functional.normalize(feat)
            ret[blockname] = feat
        return ret

    def edm_sample(self, n_sample, size, notqdm=False, num_steps=18, eta=0.0, use_amp=False):
        '''
            Sampling with stochastic sampler.
            Args:
                `n_sample`: The batch size.
                `size`: The image shape tuple (e.g. `(3, 32, 32)`).
                `num_steps`: The number of time steps for discretization. Actual NFE is `2 * num_steps - 1`.
                `eta`: controls stochasticity. Set `eta=0` for deterministic sampling.
            Returns:
                The sampled image tensor.
        '''
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        # Time steps
        times = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        times = (sigma_max ** (1 / rho) + times / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        times = torch.cat([times, torch.zeros_like(times[:1])]) # t_N = 0
        time_pairs = list(zip(times[:-1], times[1:]))

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_(x_hat, t_hat, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_next = unnormalize_to_zero_to_one(x_next)
        return x_next

    def pred_eps_(self, x, t, use_amp, clip_x=True):
        denoised = self.D_x(x, t, use_amp).to(torch.float64)
        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
        eps = (x - denoised) / t
        return eps
