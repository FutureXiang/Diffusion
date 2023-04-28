import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast

def schedules(betas, T, device, type='DDPM'):
    beta1, beta2 = betas
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), torch.linspace(beta1, beta2, T)])
    elif type == 'DDIM':
        beta_t = torch.linspace(beta1, beta2, T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}

# ===== Normalization (helper functions)

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.device = device
        self.loss = nn.MSELoss()

    def forward(self, x, use_amp=False):
        x = normalize_to_neg_one_to_one(x)

        # t ~ Uniform([1, n_T])
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        noise = torch.randn_like(x)

        sche = self.ddpm_sche
        x_t = (sche["sqrtab"][_ts, None, None, None] * x +
               sche["sqrtmab"][_ts, None, None, None] * noise)

        with autocast(enabled=use_amp):
            return self.loss(noise, self.nn_model(x_t, _ts / self.n_T))

    def get_feature(self, x, t, norm, use_amp=False):
        x = normalize_to_neg_one_to_one(x)

        _ts = torch.tensor([t]).to(self.device).repeat(x.shape[0])
        noise = torch.randn_like(x)

        sche = self.ddpm_sche
        x_t = (sche["sqrtab"][_ts, None, None, None] * x +
               sche["sqrtmab"][_ts, None, None, None] * noise)

        ret = {}
        with autocast(enabled=use_amp):
            feats = self.nn_model.get_activation(x_t, _ts / self.n_T)
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

    def sample(self, n_sample, size, notqdm=False, use_amp=False):
        sche = self.ddpm_sche
        x_i = torch.randn(n_sample, *size).to(self.device)
        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            eps = self.pred_eps_(x_i, t_is, use_amp)

            # DDPM sampling, `T` steps
            x_i = sche["oneover_sqrta"][i] * \
                    (x_i - sche["ma_over_sqrtmab"][i] * eps) + \
                  sche["sqrt_beta_t"][i] * z
                     # LET variance sigma_t = sqrt_beta_t

        x_i = unnormalize_to_zero_to_one(x_i)
        return x_i

    def ddim_sample(self, n_sample, size, steps=100, eta=0.0, notqdm=False, use_amp=False):
        def pred_x0_(x_t, eps, ab_t, clip=False):
            x_start = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip:
                x_start = torch.clip(x_start, min=-1.0, max=1.0)
            return x_start

        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            eps = self.pred_eps_(x_i, t_is, use_amp)

            # DDIM sampling, `steps` steps
            alpha = sche["alphabar_t"][time]
            x0_t = pred_x0_(x_i, eps, alpha)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        x_i = unnormalize_to_zero_to_one(x_i)
        return x_i


    def pred_eps_(self, x, t, use_amp):
        with autocast(enabled=use_amp):
            eps = self.nn_model(x, t)
        return eps
