# Modify Grad-tts

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import math
import random

from model.syncnet_hifigan import SyncNet
from model import monotonic_align
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)

from text.symbols import symbols
from utils import scheduler
import os


class FaceTTS(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters()

        self.add_blank = _config["add_blank"]
        self.n_vocab = len(symbols) + 1 if self.add_blank else len(symbols)
        self.vid_emb_dim = _config["vid_emb_dim"]
        self.n_enc_channels = _config["n_enc_channels"]
        self.filter_channels = _config["filter_channels"]
        self.filter_channels_dp = _config["filter_channels_dp"]
        self.n_heads = _config["n_heads"]
        self.n_enc_layers = _config["n_enc_layers"]
        self.enc_kernel = _config["enc_kernel"]
        self.enc_dropout = _config["enc_dropout"]
        self.window_size = _config["window_size"]
        self.n_feats = _config["n_feats"]
        self.dec_dim = _config["dec_dim"]
        self.beta_min = _config["beta_min"]
        self.beta_max = _config["beta_max"]
        self.pe_scale = _config["pe_scale"]
        self.n_spks = 2
        self.config = _config

        self.encoder = TextEncoder(
            self.n_vocab,
            self.n_feats,
            self.n_enc_channels,
            self.filter_channels,
            self.filter_channels_dp,
            self.n_heads,
            self.n_enc_layers,
            self.enc_kernel,
            self.enc_dropout,
            self.window_size,
            self.vid_emb_dim,
            self.n_spks,
        )

        self.decoder = Diffusion(
            self.n_feats,
            self.dec_dim,
            self.n_spks,
            self.vid_emb_dim,
            self.beta_min,
            self.beta_max,
            self.pe_scale,
            config=_config,
        )

        self.syncnet = SyncNet(_config)
        self.spk_fc = nn.Linear(self.vid_emb_dim, self.vid_emb_dim)
        # self.syncnet.eval()

        for p in self.syncnet.netcnnaud.parameters():
            p.requires_grad = False

        self.l1loss = nn.L1Loss()

    def relocate_input(self, x: list):
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != self.device:
                x[i] = x[i].to(self.device)
        return x

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0,
    ):
        """
        Generates mel-spectrogram from text and speaker condition (face image)
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        with torch.no_grad():
            if self.config["spk_emb"] == "speech":
                spk = self.syncnet.forward_aud(spk)
                spk = spk.squeeze(-1).detach()
                spk = torch.mean(spk, 2)
            elif self.config["spk_emb"] == "face":
                spk = self.syncnet.forward_vid(spk).squeeze(-1).detach()

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        
    
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = [
            decoder_output[:, :, :y_max_length] for decoder_output in decoder_outputs
        ]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes duration, prior, diffusion, speaker binding losses.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        spk_img = self.syncnet.forward_vid(spk)
        spk_aud = self.syncnet.forward_aud(y.unsqueeze(1))
        spk_aud = torch.mean(spk_aud, 2, keepdim=True)

        if self.config["spk_emb"] == "speech":
            spk = spk_aud.squeeze(-1)
        elif self.config["spk_emb"] == "face":
            spk = spk_img.squeeze(-1)

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)


        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask

        # Compute duration loss
        dur_loss = duration_loss(logw, logw_, x_lengths)

        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)


        # Compute diffusion loss
        diff_loss, xt, xt_hat = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        # Compute speaker loss
        spk_loss = 0.0
        out = self.syncnet.forward_perceptual(xt_hat.unsqueeze(1))
        gt_out = self.syncnet.forward_perceptual(y.unsqueeze(1))
        for i in range(2, len(out)): # you can change layers
            spk_loss += self.l1loss(out[i], gt_out[i].detach())
        spk_loss /= float(len(out))

        # Compute prior loss
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return (
            dur_loss,
            prior_loss,
            diff_loss,
            0.01 * spk_loss,
        )

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len, face = (
            batch["x"],
            batch["x_len"],
            batch["y"],
            batch["y_len"],
            batch["spk"],
        )

        (
            dur_loss,
            prior_loss,
            diff_loss,
            spk_loss,
        ) = self.compute_loss(
            x, x_len, y, y_len, spk=face, out_size=self.config["out_size"]
        )

        loss = sum(
            [
                dur_loss,
                prior_loss,
                diff_loss,
                spk_loss,
            ]
        )

        enc_grad_norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1)
        dec_grad_norm = nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1)

        self.log("train/duration_loss", dur_loss)
        self.log("train/prior_loss", prior_loss)
        self.log("train/diffusion_loss", diff_loss)
        self.log("train/spk_loss", spk_loss)
        self.log("train/total_loss", loss)

        return loss

    def training_epoch_end(self, x):
        return

    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len, face = (
            batch["x"],
            batch["x_len"],
            batch["y"],
            batch["y_len"],
            batch["spk"],
        )

        (
            dur_loss,
            prior_loss,
            diff_loss,
            spk_loss,
        ) = self.compute_loss(
            x, x_len, y, y_len, spk=face, out_size=self.config["out_size"]
        )

        loss = sum(
            [
                dur_loss,
                prior_loss,
                diff_loss,
                spk_loss,
            ]
        )

        self.log("val/duration_loss", dur_loss)
        self.log("val/prior_loss", prior_loss)
        self.log("val/diffusion_loss", diff_loss)
        self.log("val/spk_loss", spk_loss)
        self.log("val/total_loss", loss)

        return loss

    def configure_optimizers(self):
        return scheduler.set_scheduler(self)
