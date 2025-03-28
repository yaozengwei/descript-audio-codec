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

import math
from typing import Optional

import torch
from audiotools.ml import BaseModel
from torch import nn


class HiddenGenerator(BaseModel):
    """A flow matching based model that generates DAC hidden representations."""
    def __init__(
        self,
        flow_matching: nn.Module,
        dac: nn.Module,
        mel: nn.Module,
    ):
        super().__init__()
        self.flow_matching = flow_matching
        self.dac = dac
        self.mel = mel

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
    ):
        """Compute flow-matching loss
        Args:
            audio_data: (batch, time)
            sample_rate: int, optional
        """
        if sample_rate is not None:
            assert sample_rate == self.mel.sample_rate
        mel = self.mel(audio_data)

        with torch.no_grad():
            audio_data = self.dac.preprocess(audio_data, sample_rate)
            z = self.encode(audio_data)

        loss = self.flow_matching(x1=z, mel=mel)
        return loss

    @torch.inference_mode()
    @torch.no_grad()
    def infer(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
        num_steps: int = 8,
    ):
        """Flow-matching inference.
        Args:
            audio_data: (batch, time)
            sample_rate: int, optional
            num_steps: int
        """
        if sample_rate is not None:
            assert sample_rate == self.mel.sample_rate
        mel = self.mel(audio_data)

        batch, time = audio_data.shape
        z_len = math.ceil(time, self.dac.hop_length)
        noise = torch.randn(batch, self.dac.latent_dim, z_len).to(audio_data)

        z = self.flow_matching.infer(x0=noise, mel=mel, num_steps=num_steps)
        recons = self.dac.decode(z)

        return {
            "z": z,
            "audio": recons[:, :time]
        }
