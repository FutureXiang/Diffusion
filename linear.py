import argparse
from functools import partial
import os

import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ema_pytorch import EMA

from model.models import get_models_class
from utils import Config, init_seeds, reduce_tensor, gather_tensor, DataLoaderDDP, print0


def get_model(opt, load_epoch):
    if load_epoch == -1:
        load_epoch = opt.n_epoch - 1

    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type, guide=False)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    target = os.path.join(opt.save_dir, "ckpts", f"model_{load_epoch}.pth")
    print0("loading model at", target)
    checkpoint = torch.load(target, map_location=device)
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'])
    model = ema.ema_model
    model.eval()
    return model


class Classifier(nn.Module):
    def __init__(self, feat_func, blockname, dim, num_classes=10):
        super(Classifier, self).__init__()
        self.feat_func = feat_func
        self.blockname = blockname
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.feat_func(x.to(device))
            x = x[self.blockname].detach()
        return self.linear(x)


def train(opt):
    def test():
        preds = []
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            with torch.no_grad():
                lp_model.eval()
                logit = lp_model(image)
                pred = logit.argmax(dim=-1)
                preds.append(pred)
                labels.append(label.to(device))

        pred = torch.cat(preds)
        label = torch.cat(labels)
        dist.barrier()
        pred = gather_tensor(pred)
        label = gather_tensor(label)
        acc = (pred == label).sum().item() / len(label)
        return acc

    yaml_path = opt.config
    load_epoch = opt.load_epoch
    use_amp = opt.use_amp
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model = get_model(opt, load_epoch)

    epoch = opt.linear['n_epoch']
    batch_size = opt.linear['batch_size']
    base_lr = opt.linear['lrate']
    timestep = opt.linear['timestep']
    blockname = opt.linear['blockname']

    train_set = CIFAR10("./data", train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ]))
    valid_set = CIFAR10("./data", train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    train_loader, sampler = DataLoaderDDP(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader, _ = DataLoaderDDP(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
    )

    # define linear probe model
    feat_func = partial(model.get_feature, t=timestep, norm=False, use_amp=use_amp)
    with torch.no_grad():
        x = feat_func(next(iter(valid_loader))[0].to(device))
    print0("All block names:", x.keys())
    print0("Using block:", blockname)
    print0("Using timestep:", timestep)
    lp_model = Classifier(feat_func, blockname, x[blockname].shape[-1]).to(device)
    lp_model = torch.nn.parallel.DistributedDataParallel(
        lp_model, device_ids=[local_rank], output_device=local_rank)

    # train linear probe model
    loss_fn = nn.CrossEntropyLoss()
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier
    optim = torch.optim.Adam(lp_model.parameters(), lr=base_lr)
    scheduler = CosineAnnealingLR(optim, epoch)
    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            lp_model.train()
            logit = lp_model(image)
            label = label.to(device)
            loss = loss_fn(logit, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            logit = gather_tensor(logit).cpu()
            label = gather_tensor(label).cpu()

            if local_rank == 0:
                pred = logit.argmax(dim=-1)
                acc = (pred == label).sum().item() / len(label)
                nowlr = optim.param_groups[0]['lr']
                pbar.set_description("[epoch %d / iter %d]: lr %.1e loss: %.3f, acc: %.3f" % (e, i, nowlr, loss.item(), acc))
        test_acc = test()
        print0("[epoch %d]: Full Test acc: %.3f" % (e, test_acc))
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--load_epoch", type=int, default=-1)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print0(opt)

    local_rank = opt.local_rank
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank

    train(opt)
