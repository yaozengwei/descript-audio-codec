#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import math
import torch
from torch import nn


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    based on https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # x: (B, T, D)
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 Block with adaptive normalization
    based on https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        conv_kernel: int = 7,
        cond_dim: Optional[int] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert conv_kernel % 2 == 1, conv_kernel
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=cond_dim is None, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        if cond_dim is not None:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, dim, time)
            cond: (batch, cond_dim)
            mask: (batch, 1, time)

        Returns:
            x: (batch, dim, time)
        """
        residual = x

        if mask is not None:
            x = x * mask
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)

        x = self.norm(x)
        if cond is not None:
            # adaptive normalization
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
            x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.pwconv1(x)
        x = self.act(x)

        if mask is not None:
            x = x * mask.transpose(1, 2)
        x = self.grn(x)

        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + self.drop_path(x)
        return x


class ConvNeXtV2Model(nn.Module):
    """ConvNeXtV2-based model for flow-matching estimation"""
    def __init__(
        self,
        in_dim: int,
        dim: int,
        out_dim: int,
        num_layers: int,
        drop_path_rate: float = 0.0,
        use_dest_t: bool = False,
    ):
        super().__init__()
        self.use_dest_t = use_dest_t

        self.in_proj = nn.Conv1d(in_dim, dim, 1)

        self.time_embed = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim if not use_dest_t else 2 * dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(
                dim=dim,
                hidden_dim=4 * dim,
                cond_dim=dim,
                drop_path_rate=drop_path_rate,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_proj = nn.Conv1d(dim, out_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mel_embed: torch.Tensor,
        t: torch.Tensor,
        dest_t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim, time)
            mel_embed: (batch, dim, time)
            t: (batch,)
            dest_t: (batch,)
            mask: (batch, 1, time)

        Returns:
            x: (batch, out_dim, time)
        """
        x = self.in_proj(x)

        # add mel-spectrogram embedding
        assert mel_embed.shape == x.shape
        x = x + mel_embed

        if self.use_dest_t:
            assert dest_t is not None and dest_t.shape == t.shape
            time_embed = torch.cat([self.time_embed(t), self.time_embed(dest_t)], dim=-1)
        else:
            time_embed = self.time_embed(t)
        time_embed = self.time_mlp(time_embed)  # (batch, channels)

        for block in self.blocks:
            x = block(x, cond=time_embed, mask=mask)

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x)

        return x


class MelEncoder(nn.Module):
    """ConvNeXt-based mel-spectrogram encoder."""
    def __init__(
        self,
        in_dim: int,
        dim: int,
        out_dim: int,
        num_layers: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, dim, 1)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtV2Block(dim, hidden_dim=4 * dim, drop_path_rate=drop_path_rate)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_proj = nn.Conv1d(dim, out_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim, time)
            mask: (batch, 1, time)

        Returns:
            x: (batch, out_dim, time)
        """
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class FlowMatching(nn.Module):
    """Flow-matching model"""
    def __init__(
        self,
        n_mels: int = 100,
        dim: int = 512,
        num_layers: int = 12,
        mel_enc_num_layers: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.estimator = ConvNeXtV2Model(
            in_dim=dim,
            dim=dim,
            out_dim=dim,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
        )

        self.mel_encoder = MelEncoder(
            in_dim=n_mels,
            dim=dim,
            out_dim=dim,
            num_layers=mel_enc_num_layers,
            drop_path_rate=drop_path_rate,
        )

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
                nn.init.constant_(m.bias, 0)

    def compute_mel_embed(
        self,
        mel: torch.Tensor,
        time: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mel-spectrogram embedding, optionally down/up sample to a given length
        Args:
            mel: (batch, n_mels, time2)
            mask: (batch, 1, time)

        Returns:
            mel_embed: (batch, dim, time)
        """
        if mel.shape[2] != time:
            mel = nn.functional.interpolate(mel, size=time, mode='nearest')
        mel_embed = self.mel_encoder(mel, mask=mask)
        return mel_embed

    def forward(
        self,
        x1: torch.Tensor,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow-matching loss
        Args:
            x1: (batch, dim, time)
            mel: (batch, n_mels, time2), expect time2 <= time
            mask: (batch, 1, time)
        """
        mel_embed = self.compute_mel_embed(mel=mel, time=x1.shape[2], mask=mask)

        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, 1).to(x0)
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0

        vt = self.estimator(xt, mel_embed=mel_embed, t=t.squeeze(), mask=mask)

        err = ut - vt
        if mask is not None:
            loss = ((err ** 2) * mask).sum() / (mask.sum() * err.shape[1])
        else:
            loss = (err ** 2).mean()

        return loss

    @torch.no_grad()
    def infer(
        self,
        x0: torch.Tensor,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """Flow-matching inference.
        Args:
            x0: (batch, dim, time)
            mel: (batch, n_mels, time2), expect time2 <= time
            mask: (batch, 1, time)
            num_steps: int

        Returns:
            x: (batch, dim, time)
        """
        mel_embed = self.compute_mel_embed(mel=mel, time=x0.shape[2], mask=mask)

        # use fixed euler solver for ODEs.
        t_span = torch.linspace(0, 1, num_steps + 1, device=x0.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        x = x0
        batch = x.shape[0]
        for step in range(1, len(t_span)):
            vt = self.estimator(
                x,
                mel_embed=mel_embed,
                t=t.unsqueeze(0).expand(batch),
                mask=mask,
            )
            x = x + vt * dt
            t = t_span[step]

        return x


if __name__ == "__main__":
    n_mels = 100
    dim = 512
    num_layers = 12
    mel_enc_num_layers = 4
    model = FlowMatching(n_mels, dim, num_layers, mel_enc_num_layers)
    print(model)
    print("Total # of params: ", sum([p.numel() for p in model.parameters()]))

    batch = 2
    time = 200
    x = torch.randn(batch, dim, time)
    mel = torch.randn(batch, n_mels, time + 2)
    mask = torch.ones(batch, 1, time)
    loss = model(x, mel, mask)
    loss.backward()

    y = model.infer(x, mel, mask, 8)
    assert y.shape == x.shape
