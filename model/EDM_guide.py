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


class EDM_guide(nn.Module):
    def __init__(self, nn_model,
                 sigma_data, p_mean, p_std,
                 sigma_min, sigma_max, rho,
                 S_min, S_max, S_noise,
                 device, drop_prob=0.1):
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
        super(EDM_guide, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device
        self.drop_prob = drop_prob

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

    def D_x(self, x_noised, sigma, model_args, use_amp):
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
        with autocast(enabled=use_amp):
            F_x = self.nn_model(c_in * x_noised, c_noise.flatten(), *model_args)
        return c_skip * x_noised + c_out * F_x

    def forward(self, x, c, use_amp=False):
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

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        # Weighted Denoising loss
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss_4shape = weight * ((x - self.D_x(x_noised, sigma, (c, mask), use_amp)) ** 2)
        return loss_4shape.mean()


    def edm_sample(self, n_sample, size, notqdm=False, num_steps=18, eta=0.0, use_amp=False, guide_w=0.3):
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
        model_args = self.prepare_condition_(n_sample)
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
            d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_next = unnormalize_to_zero_to_one(x_next)
        return x_next

    def pred_eps_cfg(self, x, t, model_args, guide_w, use_amp):
        x_double = x.repeat(2, 1, 1, 1)
        denoised = self.D_x(x_double, t, model_args, use_amp).to(torch.float64)
        eps = (x_double - denoised) / t
        n_sample = eps.shape[0] // 2
        eps1 = eps[:n_sample]
        eps2 = eps[n_sample:]
        assert eps1.shape == eps2.shape
        eps = (1 + guide_w) * eps1 - guide_w * eps2
        return eps

    def pred_eps_(self, x, t, model_args, use_amp):
        n_sample = x.shape[0]
        denoised = self.D_x(x, t, (model_args[0][:n_sample], model_args[1][:n_sample]), use_amp).to(torch.float64)
        eps = (x - denoised) / t
        return eps

    def prepare_condition_(self, n_sample):
        n_classes = self.nn_model.num_classes
        assert n_sample % n_classes == 0
        c = torch.arange(n_classes).to(self.device)
        c = c.repeat(n_sample // n_classes)
        c = c.repeat(2)

        # 0 for conditional, 1 for unconditional
        mask = torch.zeros_like(c).to(self.device)
        mask[n_sample:] = 1.
        return c, mask
