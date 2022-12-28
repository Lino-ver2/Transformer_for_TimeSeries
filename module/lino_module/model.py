import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _generate_mask(tgt_seq: int) -> Tensor:
    inf_mask = torch.ones(tgt_seq, tgt_seq) * float('-inf')
    tgt_mask = torch.triu(inf_mask, diagonal=1)
    return tgt_mask

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
                                                d_model,
                                                nhead,
                                                batch_first=True
                                                )
        self.transformer_encoder = nn.TransformerEncoder(
                                                    encoder_layer,
                                                    num_layers=4
                                                    )

        decoder_layer = nn.TransformerDecoderLayer(
                                                d_model,
                                                nhead=8,
                                                batch_first=True
                                                )
        self.transformer_decoder = nn.TransformerDecoder(
                                                decoder_layer,
                                                num_layers=4
                                                )
        self.linear = nn.Linear(d_model, 1)

        self.positional = PositionalEncoding(d_model, dropout=0.1, max_len=5000)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # A-look ahead mask の作成
        _, tgt_seq, _ = tgt.shape
        tgt_mask = _generate_mask(tgt_seq)

        src = self.positional(src)
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        pred = self.linear(output)
        return pred


class PositionalEncoding(nn.Module):
    """Positional Encoding."""

    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        """PositionalEncodingを適用."""

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)