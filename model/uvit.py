import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import einops


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, skip=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    def __init__(self, image_shape = [3, 32, 32], embed_dim = 512,
                 patch_size = 2, depth = 12, num_heads = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chn = image_shape[0]

        self.patch_embed = PatchEmbed(
            img_size=image_shape[1:], patch_size=patch_size, in_chans=self.in_chn, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.extras = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth // 2)])

        self.mid_block = Block(dim=embed_dim, num_heads=num_heads)

        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, skip=True)
            for _ in range(depth // 2)])

        self.norm = nn.LayerNorm(embed_dim)
        self.patch_dim = patch_size ** 2 * self.in_chn
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chn, self.in_chn, 3, padding=1)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = timestep_embedding(timesteps, self.embed_dim)
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chn)
        x = self.final_layer(x)
        return x


'''
from model.uvit import UViT
net = UViT()
import torch
x = torch.zeros(1, 3, 32, 32)
t = torch.zeros(1,)

net(x, t).shape
sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

>>> 44.255328 M parameters for CIFAR-10 model
'''