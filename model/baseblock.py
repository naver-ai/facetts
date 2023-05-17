import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange


class Mish(pl.LightningModule):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(pl.LightningModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(pl.LightningModule):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(pl.LightningModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(pl.LightningModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class Residual(pl.LightningModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output
