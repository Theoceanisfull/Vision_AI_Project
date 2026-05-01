import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    x: [B, L, D]
    padding_mask: [B, L], True means "ignore this token"
    """
    if padding_mask is None:
        return x.mean(dim=1)

    valid = (~padding_mask).float().unsqueeze(-1)  # [B, L, 1]
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denom


# ------------------------------------------------------------
# Event2Vec spatial embedding
# ------------------------------------------------------------

class SpatialEmbedding(nn.Module):
    """
    Parametric spatial embedding phi(x, y, p):
      3 -> D/4 -> D/2 -> D
    with LayerNorm after each linear, and ReLU after the first two.

    Inputs x, y are expected in pixel coordinates.
    p is expected in {0, 1}.
    """

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.H = height
        self.W = width

        d1 = max(1, d_model // 4)
        d2 = max(1, d_model // 2)

        self.fc1 = nn.Linear(3, d1)
        self.ln1 = nn.LayerNorm(d1)

        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)

        self.fc3 = nn.Linear(d2, d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Normalize coordinates to [-1, 1]
        if self.W > 1:
            x_n = 2.0 * (x / (self.W - 1)) - 1.0
        else:
            x_n = torch.zeros_like(x)

        if self.H > 1:
            y_n = 2.0 * (y / (self.H - 1)) - 1.0
        else:
            y_n = torch.zeros_like(y)

        # Map polarity from {0,1} -> {-1,1}
        p_n = 2.0 * p - 1.0

        feat = torch.stack([x_n, y_n, p_n], dim=-1)  # [B, L, 3]

        feat = self.fc1(feat)
        feat = self.ln1(feat)
        feat = F.relu(feat)

        feat = self.fc2(feat)
        feat = self.ln2(feat)
        feat = F.relu(feat)

        feat = self.fc3(feat)
        feat = self.ln3(feat)
        return feat


# ------------------------------------------------------------
# Event2Vec temporal embedding
# ------------------------------------------------------------

class TemporalEmbedding(nn.Module):
    """
    Temporal embedding over Δt:
      Conv1d 1 -> D/4 -> D/2 -> D
    with kernel size 3, stride 1, padding 1.

    Note:
    The paper text says "depth-wise convolutional layers", while the
    figure describes Conv1d channel expansion 1->D/4->D/2->D.
    This implementation uses standard Conv1d so it is runnable and
    matches the channel sizes shown in the paper figure.
    """

    def __init__(self, d_model: int):
        super().__init__()
        d1 = max(1, d_model // 4)
        d2 = max(1, d_model // 2)

        self.conv1 = nn.Conv1d(1, d1, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(d1)

        self.conv2 = nn.Conv1d(d1, d2, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(d2)

        self.conv3 = nn.Conv1d(d2, d_model, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(d_model)

    @staticmethod
    def _apply_conv_ln_act(x: torch.Tensor, conv: nn.Module, ln: nn.Module, act: bool) -> torch.Tensor:
        # x: [B, C, L]
        x = conv(x)                  # [B, C_out, L]
        x = x.transpose(1, 2)        # [B, L, C_out]
        x = ln(x)
        if act:
            x = F.relu(x)
        x = x.transpose(1, 2)        # [B, C_out, L]
        return x

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        dt: [B, L]
        Returns: [B, L, D]
        """
        x = dt.unsqueeze(1)  # [B, 1, L]

        x = self._apply_conv_ln_act(x, self.conv1, self.ln1, act=True)
        x = self._apply_conv_ln_act(x, self.conv2, self.ln2, act=True)
        x = self._apply_conv_ln_act(x, self.conv3, self.ln3, act=False)

        return x.transpose(1, 2)  # [B, L, D]


# ------------------------------------------------------------
# Event2Vec encoder
# ------------------------------------------------------------

class Event2Vec(nn.Module):
    """
    Input events are [B, L, 5] with columns:
      x, y, t, p, rho

    - x, y: pixel coordinates
    - t: normalized timestamp in [0, 1] per sample (recommended)
    - p: polarity in {0, 1}
    - rho: intensity count, use 1.0 for raw/native events

    Eq. (6):
      V[i] = (log(rho[i]) + 1) * (Embed_s(x,y,p) + Embed_t(Δt)[i])
    """

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.spatial = SpatialEmbedding(d_model, height, width)
        self.temporal = TemporalEmbedding(d_model)

    def forward(self, events: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        events: [B, L, 5]
        padding_mask: [B, L], True means "ignore this token"
        """
        events = events.float()
        x, y, t, p, rho = events.unbind(dim=-1)

        # Δt[i] = t[i+1] - t[i], last value = 0
        dt = torch.zeros_like(t)
        dt[:, :-1] = t[:, 1:] - t[:, :-1]

        vs = self.spatial(x, y, p)
        vt = self.temporal(dt)

        scale = (torch.log(rho.clamp_min(1.0)) + 1.0).unsqueeze(-1)
        v = scale * (vs + vt)

        if padding_mask is not None:
            v = v.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return v


# ------------------------------------------------------------
# Shared bidirectional attention block
# ------------------------------------------------------------

class SharedBidirectionalAttentionBlock(nn.Module):
    """
    Practical shared bidirectional attention block:
    - same attention weights are used on forward and reversed sequence
    - outputs are fused with a linear layer
    - FFN follows standard Transformer style
    - optional average-pooling along sequence length
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        pool: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2) if pool else None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # Shared forward / backward attention
        y = self.norm1(x)

        fwd, _ = self.attn(
            y, y, y,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        y_rev = torch.flip(y, dims=[1])
        mask_rev = torch.flip(padding_mask, dims=[1]) if padding_mask is not None else None

        bwd_rev, _ = self.attn(
            y_rev, y_rev, y_rev,
            key_padding_mask=mask_rev,
            need_weights=False,
        )
        bwd = torch.flip(bwd_rev, dims=[1])

        x = x + self.drop1(self.fuse(torch.cat([fwd, bwd], dim=-1)))
        x = x + self.drop2(self.ffn(self.norm2(x)))

        # Optional sequence pooling
        if self.pool is not None:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)

            if padding_mask is not None:
                valid = (~padding_mask).float().unsqueeze(1)   # [B, 1, L]
                valid = self.pool(valid).squeeze(1)            # [B, L/2]
                padding_mask = valid < 0.5

        return x, padding_mask


# ------------------------------------------------------------
# Full classifier
# ------------------------------------------------------------

class Event2VecClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        height: int,
        width: int,
        d_model: int = 64,
        depth: int = 4,
        num_heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        pool_after_each_block: Union[bool, Sequence[bool]] = False,
    ):
        super().__init__()
        self.event2vec = Event2Vec(d_model=d_model, height=height, width=width)

        if isinstance(pool_after_each_block, bool):
            pool_flags = [pool_after_each_block] * depth
        else:
            pool_flags = list(pool_after_each_block)
            assert len(pool_flags) == depth, "pool_after_each_block must have length == depth"

        self.blocks = nn.ModuleList([
            SharedBidirectionalAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                pool=pool_flags[i],
            )
            for i in range(depth)
        ])

        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        events: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """
        events: [B, L, 5] -> columns [x, y, t, p, rho]
        padding_mask: [B, L], True means ignore / padded token
        """
        x = self.event2vec(events, padding_mask=padding_mask)

        for block in self.blocks:
            x, padding_mask = block(x, padding_mask)

        features = masked_mean(x, padding_mask)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits


# ------------------------------------------------------------
# Suggested configs from the paper appendix
# ------------------------------------------------------------

EVENT2VEC_CONFIGS = {
    "dvs_gesture": dict(
        d_model=64,
        ffn_dim=128,
        num_heads=2,
        depth=4,
        pool_after_each_block=True,   # paper uses pooling after each FFN for DVS Gesture
    ),
    "asl_dvs": dict(
        d_model=64,
        ffn_dim=128,
        num_heads=2,
        depth=2,
        pool_after_each_block=False,
    ),
    "dvs_lip": dict(
        d_model=192,
        ffn_dim=384,
        num_heads=6,
        depth=16,
        pool_after_each_block=False,
    ),
}


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example for DVS Gesture (11 classes, 128x128 sensor)
    cfg = EVENT2VEC_CONFIGS["dvs_gesture"]

    model = Event2VecClassifier(
        num_classes=11,
        height=128,
        width=128,
        **cfg,
    )

    # Dummy batch:
    # columns = [x, y, t, p, rho]
    # x, y in pixel coordinates
    # t normalized to [0, 1]
    # p in {0, 1}
    # rho = 1 for raw/native events
    B, L = 2, 4096
    events = torch.zeros(B, L, 5)

    events[..., 0] = torch.randint(0, 128, (B, L)).float()                 # x
    events[..., 1] = torch.randint(0, 128, (B, L)).float()                 # y
    events[..., 2] = torch.sort(torch.rand(B, L), dim=1).values            # t
    events[..., 3] = torch.randint(0, 2, (B, L)).float()                   # p
    events[..., 4] = 1.0                                                    # rho

    # Optional padding mask: True = padded token
    padding_mask = None

    logits = model(events, padding_mask=padding_mask)
    print("logits shape:", logits.shape)  # [B, num_classes]