import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torchvision.utils import make_grid, save_image
from ema_pytorch import EMA

from model.models import get_models_class

# ===== Config yaml files (helper functions)

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])


def get_default_steps(model_type, steps):
    if steps is not None:
        return steps
    else:
        return {'DDPM': 100, 'EDM': 18}[model_type]

# ===== Multi-GPU training (helper functions)

def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list

# ===== sampling =====

def sample(opt):
    print(opt)
    yaml_path = opt.config
    local_rank = opt.local_rank
    mode = opt.mode
    steps = opt.steps
    eta = opt.eta
    batches = opt.batches
    use_ema = opt.ema
    ep = opt.epoch
    w = opt.w

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)
    if ep == -1:
        ep = opt.n_epoch - 1

    device = "cuda:%d" % local_rank
    steps = get_default_steps(opt.model_type, steps)
    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type, guide=True)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     drop_prob=0.1)
    diff.to(device)

    target = os.path.join(opt.save_dir, "ckpts", f"model_{ep}.pth")
    print("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    if use_ema:
        ema = EMA(diff, beta=0.9999, update_after_step=1000, update_every=10)
        ema.to(device)
        ema.load_state_dict(checkpoint['EMA'])
        model = ema.ema_model
        prefix = "EMA"
    else:
        diff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(diff)
        diff = torch.nn.parallel.DistributedDataParallel(
            diff, device_ids=[local_rank], output_device=local_rank)
        diff.load_state_dict(checkpoint['MODEL'])
        model = diff.module
        prefix = ""
    model.eval()

    if local_rank == 0:
        if opt.model_type == 'EDM':
            gen_dir = os.path.join(opt.save_dir, f"{prefix}generated_ep{ep}_w{w}_edm_steps{steps}_eta{eta}")
        else:
            if mode == 'DDPM':
                gen_dir = os.path.join(opt.save_dir, f"{prefix}generated_ep{ep}_w{w}_ddpm")
            else:
                gen_dir = os.path.join(opt.save_dir, f"{prefix}generated_ep{ep}_w{w}_ddim_steps{steps}_eta{eta}")
        os.makedirs(gen_dir)
        gen_dir_png = os.path.join(gen_dir, "pngs")
        os.makedirs(gen_dir_png)
        res = []

    for batch in range(batches):
        with torch.no_grad():
            assert 400 % dist.get_world_size() == 0
            samples_per_process = 400 // dist.get_world_size()
            args = dict(n_sample=samples_per_process, size=opt.network['image_shape'], guide_w=w, notqdm=(local_rank != 0))
            if opt.model_type == 'EDM':
                x_gen = model.edm_sample(**args, num_steps=steps, eta=eta)
            else:
                if mode == 'DDPM':
                    x_gen = model.sample(**args)
                else:
                    x_gen = model.ddim_sample(**args, steps=steps, eta=eta)
        dist.barrier()
        x_gen = gather_tensor(x_gen)
        if local_rank == 0:
            res.append(x_gen)
            grid = make_grid(x_gen.cpu(), nrow=20)
            png_path = os.path.join(gen_dir, f"grid_{batch}.png")
            save_image(grid, png_path)

    if local_rank == 0:
        res = torch.cat(res)
        for no, img in enumerate(res):
            png_path = os.path.join(gen_dir_png, f"{no}.png")
            save_image(img, png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--mode", type=str, choices=['DDPM', 'DDIM'], default='DDIM')
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--batches", type=int, default=125)
    parser.add_argument("--ema", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--w", type=float, default=0.3)
    opt = parser.parse_args()

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    sample(opt)
