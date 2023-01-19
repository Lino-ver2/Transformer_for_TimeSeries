import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Tuple


def training(model: object,
             train: DataLoader,
             test: DataLoader,
             device: torch.device,
             criterion: object,
             optimizer: object,
             epochs: int,
             verbose=10,
             center=80) -> Tuple[object, Tensor, Tensor]:
    """訓練用関数"""
    train_loss = []
    validation_loss = []
    test_loss = []
    print(' start '.center(center, '-'))
    start_point = time.time()
    for epoch in range(epochs):
        train_epoch_loss = []
        validation_epoch_loss = []

        cache = None
        for i, pack in enumerate(train):
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
                loss = criterion(output.squeeze(), cached_y)
                train_epoch_loss.append(loss.item())
                # 勾配計算
                loss.backward()
                optimizer.step()
            # モデル評価
            model.eval()
            output = model(src, tgt)
            loss = criterion(output.squeeze(), y)
            validation_epoch_loss.append(loss.item())
            # データをキャッシュに保存して次回の訓練データにする
            cache = (src, tgt, y)

        test_epoch_loss = []
        for pack in test:
            src, tgt, y = [content.to(device) for content in pack]
            model.eval()
            output = model(src, tgt)
            loss = criterion(output.reshape(-1), y.reshape(-1))
            test_epoch_loss.append(loss.item())

        validation_loss.append(validation_epoch_loss)
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)

        if epoch % verbose == 0:
            print(f' epoch_{epoch} '.center(center))
            train_mean = torch.mean(torch.tensor(train_epoch_loss)).item()
            valid_mean = torch.mean(torch.tensor(validation_epoch_loss)).item()
            test_mean = torch.mean(torch.tensor(test_epoch_loss)).item()
            print('train_loss: ', round(train_mean, 4),
                  '| validation_loss: ', round(valid_mean, 4),
                  '| test_loss: ', round(test_mean, 4))

    print(' complete!! '.center(center, '-'))
    print(f'Execution_time: {round(time.time() - start_point, 3)}')
    packs = (train_loss, validation_loss, test_loss)
    loss_pack = [torch.tensor(loss) for loss in packs]
    return model, loss_pack
