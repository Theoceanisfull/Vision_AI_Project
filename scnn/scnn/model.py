from __future__ import annotations

from dataclasses import dataclass

import snntorch as snn
import snntorch.utils as snn_utils
import torch
import torch.nn as nn
from snntorch import surrogate

from .config import ModelConfig


def build_surrogate(name: str, kwargs: dict) -> callable:
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(**kwargs)
    if name == "atan":
        return surrogate.atan(**kwargs)
    if name == "sigmoid":
        return surrogate.sigmoid(**kwargs)
    if name == "triangular":
        return surrogate.triangular(**kwargs)
    if name == "spike_rate_escape":
        return surrogate.spike_rate_escape(**kwargs)
    if name == "straight_through_estimator":
        return surrogate.straight_through_estimator()
    raise ValueError(f"Unsupported surrogate type: {name}")


@dataclass
class ConvShape:
    h: int
    w: int


def _conv_out_size(size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return ((size + 2 * padding - kernel_size) // stride) + 1


class ConvSNN(nn.Module):
    """Configurable Conv-SNN for ASL-DVS encoded input [B, T, C, H, W]."""

    def __init__(
        self,
        cfg: ModelConfig,
        sensor_size: tuple[int, int] = (180, 240),
        *,
        conv_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.sensor_h, self.sensor_w = sensor_size

        spike_grad = build_surrogate(cfg.surrogate, cfg.surrogate_kwargs)
        k = cfg.kernel_size
        pad = k // 2

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_ch = cfg.input_channels
        shape = ConvShape(self.sensor_h, self.sensor_w)

        for out_ch in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False))
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
            self.lif_layers.append(
                snn.Leaky(
                    beta=cfg.beta,
                    threshold=cfg.threshold,
                    spike_grad=spike_grad,
                    init_hidden=True,
                )
            )
            self.pool_layers.append(nn.MaxPool2d(kernel_size=cfg.pool_kernel))

            shape = ConvShape(
                _conv_out_size(shape.h, k, padding=pad),
                _conv_out_size(shape.w, k, padding=pad),
            )
            shape = ConvShape(
                _conv_out_size(shape.h, cfg.pool_kernel, stride=cfg.pool_kernel),
                _conv_out_size(shape.w, cfg.pool_kernel, stride=cfg.pool_kernel),
            )
            in_ch = out_ch

        flatten_dim = in_ch * shape.h * shape.w

        self.drop = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(flatten_dim, cfg.hidden_size)
        self.lif_fc = snn.Leaky(
            beta=cfg.beta,
            threshold=cfg.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
        )

        self.fc_out = nn.Linear(cfg.hidden_size, cfg.num_classes)
        self.lif_out = snn.Leaky(
            beta=cfg.beta,
            threshold=cfg.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

    def reset_hidden(self) -> None:
        snn_utils.reset(self)

    def forward_step(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_t
        for conv, bn, lif, pool in zip(self.conv_layers, self.bn_layers, self.lif_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = lif(x)
            x = pool(x)

        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.lif_fc(x)

        x = self.fc_out(x)
        spk_out, mem_out = self.lif_out(x)
        return spk_out, mem_out

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_first: bool = False,
        reset: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            return self.forward_step(x)

        if x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")

        if reset:
            self.reset_hidden()

        if not time_first:
            x = x.transpose(0, 1)

        spk_rec = []
        mem_rec = []
        for t in range(x.size(0)):
            spk_t, mem_t = self.forward_step(x[t])
            spk_rec.append(spk_t)
            mem_rec.append(mem_t)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


def build_model(cfg: ModelConfig, sensor_size: tuple[int, int] = (180, 240)) -> ConvSNN:
    if cfg.architecture == "base":
        channels = tuple(cfg.conv_channels)
    elif cfg.architecture == "deep":
        channels = tuple(cfg.deep_conv_channels)
    else:
        raise ValueError(f"Unsupported architecture: {cfg.architecture}")

    if len(channels) < 2:
        raise ValueError("Model requires at least 2 convolutional stages")

    return ConvSNN(cfg, sensor_size=sensor_size, conv_channels=channels)


__all__ = ["ConvSNN", "build_surrogate", "build_model"]
