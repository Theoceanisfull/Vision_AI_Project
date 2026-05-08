"""
Sign Language Translation model.

Architecture:
  KeypointEncoder  : linear projection -> positional encoding -> Transformer encoder
                     -> LayerNorm -> output projected to mBART hidden dim (1024)
  mBART decoder    : facebook/mbart-large-cc25, text encoder frozen,
                     decoder fine-tuned with visual features as cross-attention input
"""

import math

import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.modeling_outputs import BaseModelOutput


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class KeypointEncoder(nn.Module):
    """Transformer encoder that maps keypoint sequences to mBART's hidden space."""

    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward, dropout, decoder_hidden):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        # LayerNorm before projecting to decoder space — aligns statistics with
        # what mBART's cross-attention layers were trained to receive.
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, decoder_hidden)

    def forward(self, src, padding_mask=None):
        """
        src          : (B, T, input_dim)
        padding_mask : (B, T) — True = ignore (padding positions)
        returns      : (B, T, decoder_hidden)
        """
        x = self.pos_enc(self.input_proj(src))
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.output_proj(self.norm(x))


class SignTranslationModel(nn.Module):
    """KeypointEncoder + mBART-large-cc25 decoder."""

    def __init__(self, cfg):
        super().__init__()
        mc  = cfg['model']
        self.mbart      = MBartForConditionalGeneration.from_pretrained(mc['decoder_model'])
        dec_hidden      = self.mbart.config.d_model    # 1024

        self.encoder = KeypointEncoder(
            input_dim       = mc['keypoint_dim'],
            d_model         = mc['encoder_d_model'],
            nhead           = mc['encoder_nhead'],
            num_layers      = mc['encoder_layers'],
            dim_feedforward = mc['encoder_ff_dim'],
            dropout         = mc['dropout'],
            decoder_hidden  = dec_hidden,
        )

        # Freeze mBART text encoder — we replace it with our visual encoder
        for p in self.mbart.model.encoder.parameters():
            p.requires_grad = False

        self.forced_bos_token_id = None   # set in train.py after tokenizer is built

    # ------------------------------------------------------------------
    def _encode(self, keypoints, padding_mask):
        enc_out   = self.encoder(keypoints, padding_mask)
        attn_mask = (~padding_mask).long()   # mBART: 1=attend, 0=ignore
        return enc_out, attn_mask

    # ------------------------------------------------------------------
    def forward(self, keypoints, padding_mask, labels, **_):
        enc_out, attn_mask = self._encode(keypoints, padding_mask)
        out = self.mbart(
            input_ids       = None,
            attention_mask  = attn_mask,
            encoder_outputs = BaseModelOutput(last_hidden_state=enc_out),
            labels          = labels,
        )
        return out.loss, out.logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, keypoints, padding_mask, num_beams=4, max_new_tokens=128, **_):
        enc_out, attn_mask = self._encode(keypoints, padding_mask)
        return self.mbart.generate(
            encoder_outputs      = BaseModelOutput(last_hidden_state=enc_out),
            attention_mask       = attn_mask,
            forced_bos_token_id  = self.forced_bos_token_id,
            num_beams            = num_beams,
            max_new_tokens       = max_new_tokens,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            repetition_penalty   = 1.3,   # penalise repeating generic phrases
        )


class S3DSignTranslationModel(nn.Module):
    """
    S3D features (T', 1024) -> lightweight Transformer -> mBART decoder.

    S3D already provides rich motion representations, so we only need a
    shallow context encoder on top before handing off to mBART.
    """

    def __init__(self, cfg):
        super().__init__()
        mc         = cfg['model']
        self.mbart = MBartForConditionalGeneration.from_pretrained(mc['decoder_model'])
        dec_hidden = self.mbart.config.d_model   # 1024

        d_model = mc.get('encoder_d_model', dec_hidden)

        # Optional input projection (identity when d_model == S3D feature dim)
        feat_dim        = mc['keypoint_dim']   # 1024 for S3D
        self.input_proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()
        self.pos_enc    = PositionalEncoding(d_model, mc['dropout'])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=mc['encoder_nhead'],
            dim_feedforward=mc['encoder_ff_dim'],
            dropout=mc['dropout'], batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=mc['encoder_layers'], enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, dec_hidden) if d_model != dec_hidden else nn.Identity()

        # Freeze mBART text encoder
        for p in self.mbart.model.encoder.parameters():
            p.requires_grad = False

        self.forced_bos_token_id = None

    def _encode(self, keypoints, padding_mask):
        x         = self.pos_enc(self.input_proj(keypoints))
        x         = self.transformer(x, src_key_padding_mask=padding_mask)
        enc_out   = self.output_proj(self.norm(x))
        attn_mask = (~padding_mask).long()
        return enc_out, attn_mask

    def forward(self, keypoints, padding_mask, labels, **_):
        enc_out, attn_mask = self._encode(keypoints, padding_mask)
        out = self.mbart(
            input_ids       = None,
            attention_mask  = attn_mask,
            encoder_outputs = BaseModelOutput(last_hidden_state=enc_out),
            labels          = labels,
        )
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, keypoints, padding_mask, num_beams=4, max_new_tokens=128, **_):
        enc_out, attn_mask = self._encode(keypoints, padding_mask)
        return self.mbart.generate(
            encoder_outputs      = BaseModelOutput(last_hidden_state=enc_out),
            attention_mask       = attn_mask,
            forced_bos_token_id  = self.forced_bos_token_id,
            num_beams            = num_beams,
            max_new_tokens       = max_new_tokens,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            repetition_penalty   = 1.3,
        )


# ------------------------------------------------------------------
def build_model(cfg):
    if cfg['model'].get('feature_type') == 's3d':
        return S3DSignTranslationModel(cfg)
    return SignTranslationModel(cfg)


def build_tokenizer(cfg):
    tok          = MBart50TokenizerFast.from_pretrained(cfg['model']['decoder_model'])
    tok.tgt_lang = cfg['model'].get('tgt_lang', 'en_XX')
    return tok
