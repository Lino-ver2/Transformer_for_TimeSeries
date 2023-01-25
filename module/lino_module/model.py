import math

import torch
import torch.nn as nn

from typing import Callable, Optional
from torch import Tensor


class WithAuxiliary(nn.Module):
    def __init__(self, base_model, auxiliary_model):
        super(WithAuxiliary, self).__init__()
        self.base: Callable[[Tensor], Tensor] = base_model
        self.auxiliary: Callable[[Tensor], Tensor] = auxiliary_model

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                y: Optional[Tensor] = None) -> Tensor:
        base_pred = self.base(src, tgt)
        auxiliary_pred = self.base(src, tgt)
        if self.training:
            auxiliary_label = y - base_pred.squeeze(2)
            return base_pred, auxiliary_pred, auxiliary_label
        else:
            return base_pred, auxiliary_pred


class TransformerModel(nn.Module):
    """Trasnsormer for Time Series
        参考論文: https://arxiv.org/abs/2001.08317
    """
    def __init__(self, d_model: int, nhead: int, device):
        super(TransformerModel, self).__init__()

        self.positional = PositionalEncoding(d_model)
        self.device = device

        encoder_layer = nn.TransformerEncoderLayer(
                                                d_model,
                                                nhead,
                                                dropout=0.2,
                                                batch_first=True
                                                )
        self.transformer_encoder = nn.TransformerEncoder(
                                                    encoder_layer,
                                                    num_layers=4
                                                    )

        decoder_layer = nn.TransformerDecoderLayer(
                                                d_model,
                                                nhead,
                                                dropout=0.2,
                                                batch_first=True
                                                )
        self.transformer_decoder = nn.TransformerDecoder(
                                                decoder_layer,
                                                num_layers=4
                                                )
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # Decoder用のtgt_maskを作成
        _, tgt_seq, _ = tgt.shape
        tgt_mask = _generate_mask(tgt_seq)   # A-look ahead mask

        # Positional Encoding
        src = self.positional(src)
        # Encoder
        memory = self.transformer_encoder(src)
        # Decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask.to(self.device))
        # 線形変で出力の形状へ
        pred = self.linear(output)
        return pred


def _generate_mask(tgt_seq: int) -> Tensor:
    """デコーダ入力用の Self Attention用のマスクを作成"""
    inf_mask = torch.ones(tgt_seq, tgt_seq) * float('-inf')
    tgt_mask = torch.triu(inf_mask, diagonal=1)
    return tgt_mask


class PositionalEncoding(nn.Module):
    """PositionalEnoder"""
    def __init__(self, d_model: int, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0), :]
        return x


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, t):
        return torch.sqrt(self.mse(y, t))


class LossWithAuxiliary(nn.Module):
    def __init__(self, base_func, auxiliary_func):
        super(LossWithAuxiliary, self).__init__()
        self.base_func: Callable[[Tensor], Tensor] = base_func
        self.auxiliary_func: Callable[[Tensor], Tensor] = auxiliary_func

    def forward(self, base_pred: Tensor, auxiliary_pred: Tensor,
                y: Tensor, auxiliary_label: Tensor) -> Tensor:
        base_loss = self.base_func(base_pred, y)
        auxiliary_loss = self.auxiliary_func(auxiliary_pred, auxiliary_label)
        loss = base_loss * 0.5 + auxiliary_loss * 0.5
        return loss, base_loss, auxiliary_loss
