import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ema_pytorch import EMA

from model.DDPM_guide import DDPM
from model.unet_guide import UNet

# ===== Visualize (helper functions)

def get_real_samples_grid(x, c, size, opt):
    # append some real images at bottom, order by class also
    x_real = torch.Tensor(size)
    for k in range(opt.n_classes):
        sample_label_k = torch.squeeze((c == k).nonzero())
        n_sample_per_class = opt.n_sample // opt.n_classes
        for j in range(n_sample_per_class):
            try:
                idx = sample_label_k[j]
            except:
                idx = 0
            x_real[k + (j * opt.n_classes)] = x[idx]
    return x_real

# ===== Config yaml files (helper functions)

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def DataLoaderDDP(dataset, batch_size, shuffle=True, **kwargs):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
    )
    return dataloader, sampler

# ===== training =====

def train(opt):
    yaml_path = opt.config
    local_rank = opt.local_rank

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    device = "cuda:%d" % local_rank
    ddpm = DDPM(nn_model=UNet(**opt.unet),
                **opt.ddpm,
                device=device,
                drop_prob=0.1)
    ddpm.to(device)
    if local_rank == 0:
        ema = EMA(ddpm, beta=0.9999, update_after_step=1000, update_every=10)
        ema.to(device)

    ddpm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ddpm)
    ddpm = torch.nn.parallel.DistributedDataParallel(
        ddpm, device_ids=[local_rank], output_device=local_rank)

    tf = [transforms.ToTensor()]
    if opt.flip:
        tf = [transforms.RandomHorizontalFlip()] + tf
    tf = transforms.Compose(tf)
    train_set = CIFAR10("./data", train=True, download=False, transform=tf)
    print("CIFAR10 train dataset:", len(train_set))

    train_loader, sampler = DataLoaderDDP(train_set,
                                          batch_size=opt.batch_size,
                                          shuffle=True)

    lr = opt.lrate
    DDP_multiplier = dist.get_world_size()
    print("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    sched = CosineAnnealingLR(optim, opt.n_epoch)


    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        ddpm.load_state_dict(checkpoint['DDPM'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        optim.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])

    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        sampler.set_epoch(ep)
        dist.barrier()
        # training
        ddpm.train()
        if local_rank == 0:
            now_lr = optim.param_groups[0]['lr']
            print(f'epoch {ep}, lr {now_lr:f}')
            loss_ema = None
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=ddpm.parameters(), max_norm=1.0)
            optim.step()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            if local_rank == 0:
                ema.update()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")

        # cosine scheduler
        sched.step()

        # testing
        if local_rank == 0:
            if ep % 50 == 0 or ep == opt.n_epoch - 1:
                pass
            else:
                continue

            ddpm.eval()
            with torch.no_grad():
                x_gen = ddpm.module.ddim_sample(opt.n_sample, x.shape[1:], guide_w=opt.w, steps=100, eta=0.0)
            # save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            x_real = get_real_samples_grid(x, c, x_gen.shape, opt)
            x_all = torch.cat([x_gen.cpu(), x_real])
            grid = make_grid(x_all, nrow=10)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_w{opt.w}.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

            ema.ema_model.eval()
            with torch.no_grad():
                x_gen = ema.ema_model.ddim_sample(opt.n_sample, x.shape[1:], guide_w=opt.w, steps=100, eta=0.0)
            # save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            x_real = get_real_samples_grid(x, c, x_gen.shape, opt)
            x_all = torch.cat([x_gen.cpu(), x_real])
            grid = make_grid(x_all, nrow=10)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_w{opt.w}_ema.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

            # optionally save model
            if opt.save_model:
                checkpoint = {
                    'DDPM': ddpm.state_dict(),
                    'EMA': ema.state_dict(),
                    'opt': optim.state_dict(),
                    'sched': sched.state_dict(),
                }
                save_path = os.path.join(model_dir, f"model_{ep}.pth")
                torch.save(checkpoint, save_path)
                print('saved model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    opt = parser.parse_args()

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)
