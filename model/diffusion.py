import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from model.baseblock import (
    ResnetBlock,
    Residual,
    Block,
    Mish,
    Upsample,
    Downsample,
    Rezero,
    LinearAttention,
)


class PosEmbedding(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(pl.LightningModule):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        multi_spks=1,
        spk_emb_dim=512,
        n_feats=80,
        pe_scale=1000,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.multi_spks = multi_spks
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if self.multi_spks:
            self.spk_mlp = nn.Sequential(
                nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                nn.Linear(spk_emb_dim * 4, n_feats),
            )

        self.time_pos_emb = PosEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if self.multi_spks else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_block = Block(dim, dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if not self.multi_spks:
            x = torch.stack([mu, x], 1)

        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.size(-1))
            x = torch.stack([mu, x, s], 1)

        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        n_feats,
        dim,
        multi_spks=1,
        spk_emb_dim=512,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
        config=dict(),
    ):
        super().__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.multi_spks = multi_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.config = config

        self.estimator = GradLogPEstimator2d(
            dim,
            multi_spks=multi_spks,
            spk_emb_dim=spk_emb_dim,
            pe_scale=pe_scale,
            n_feats=self.n_feats,
        )

    def get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
        else:
            noise = beta_init + (beta_term - beta_init) * t
        return noise

    def forward_diff(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self.get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (
            1.0 - torch.exp(-0.5 * cum_noise)
        )
        var = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(
            x0.shape, dtype=x0.dtype, device=self.device, requires_grad=False
        )

        xt = mean + z * torch.sqrt(var)

        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diff(self, z, mask, mu, n_steps, stoc=False, spk=None):
        h = 1.0 / n_steps
        xt = z * mask
        xts = []
        xts.append(xt)
        for i in range(n_steps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.size(0), dtype=z.dtype, device=self.device
            )
            time = t.unsqueeze(-1).unsqueeze(1)

            noise_t = self.get_noise(
                time, self.beta_min, self.beta_max, cumulative=False
            )

            if stoc:
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det *= noise_t * h
                dxt_stoc = torch.randn(
                    z.size(), dtype=z.dtype, device=self.device, requires_grad=False
                )
                dxt_stoc *= torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc

            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt *= noise_t * h

            xt = (xt - dxt) * mask
            xts.append(xt)

        return xts

    @torch.no_grad()
    def forward(self, z, mask, mu, n_steps, stoc=False, spk=None):
        return self.reverse_diff(z, mask, mu, n_steps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diff(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self.get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_t = self.estimator(xt, mask, mu, t, spk)
        pred_noise = noise_t * torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((pred_noise + z) ** 2) / (torch.sum(mask) * self.n_feats)
        if self.config["perceptual_loss"]:
            dxt = 0.5 * (mu - xt - noise_t)
            dxt *= cum_noise
            xt_hat = (xt - dxt) * mask
            return loss, xt, xt_hat
        else:
            return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(
            x0.size(0), dtype=x0.dtype, device=self.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)
