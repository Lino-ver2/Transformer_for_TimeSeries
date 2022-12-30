import time
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class TransformerModel(nn.Module):
    """Trasnsormer for Time Series
        参考論文: https://arxiv.org/abs/2001.08317
    """
    def __init__(self, d_model: int, nhead: int):
        super(TransformerModel, self).__init__()

        self.positional = PositionalEncoding(d_model)

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
                                                nhead=8,
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
        output = self.transformer_decoder(tgt, memory, tgt_mask)
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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0), :]
        return x


def training(model: object,
             dataset: DataLoader,
             device: torch.device,
             criterion: object,
             optimizer: object,
             epochs: int,
             verbose=10,
             center=80) -> Tuple[object, Tensor, Tensor]:
    train_loss = []
    validation_loss = []
    print(' start '.center(center, '-'))
    start_point = time.time()
    for epoch in range(epochs):
        epoch_point = time.time()
        train_epoch_loss = []
        validation_epoch_loss = []

        cache = None
        for i, pack in enumerate(dataset):
            src, tgt, y = [content.to(device) for content in pack]
            # モデル訓練
            if i == 0:
                pass
            else:
                # キャッシュから１バッチ前のデータで訓練
                cached_src, cached_tgt, cached_y = cache
                model.train()
                optimizer.zero_grad()
                output = model(cached_src, cached_tgt)
                loss = criterion(output[:, 1, :], cached_y)
                train_epoch_loss.append(loss.item())
                # 勾配計算
                loss.backward()
                optimizer.step()
            # モデル評価
            model.eval()
            output = model(src, tgt)
            loss = criterion(output[:, 1, :], y)
            validation_epoch_loss.append(loss.item())
            # データをキャッシュに保存して次回の訓練データにする
            cache = (src, tgt, y)

        validation_loss.append(validation_epoch_loss)
        train_loss.append(train_epoch_loss)

        if epoch % verbose == 0:
            print(f' epoch_{epoch} '.center(center))
            train_mean = torch.mean(torch.tensor(train_epoch_loss)).item()
            test_mean = torch.mean(torch.tensor(validation_epoch_loss)).item()
            print('train_loss: ', round(train_mean, 4),
                  '| validation_loss: ', round(test_mean, 4),
                  '| time: ', round(time.time() - epoch_point, 3))

    print(' complete!! '.center(center, '-'))
    print(f'Execution_time: {round(time.time() - start_point, 3)}')
    return model, torch.tensor(train_loss), torch.tensor(validation_loss)
