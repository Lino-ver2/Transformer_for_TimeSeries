import time

import torch

from typing import List, Dict, Tuple, Callable, Optional
from torch import Tensor
from torch.utils.data import DataLoader


def training_with_auxiliary(model: Callable[[Tensor], Tensor],
                            train: DataLoader,
                            test: DataLoader,
                            device: torch.device,
                            criterion: Callable[[Tensor], Tensor],
                            optimizer: object,
                            epochs: int,
                            verbose=10,
                            center=80) -> Tuple[object, Tensor, Tensor]:
    """補助モデルを取り入れた訓練用関数"""
    loss_pack = {'train': {'loss': [], 'base': [],
                           'auxiliary': [], 'auxiliary_rate': []},
                 'eval': {'loss': [], 'base': [],
                          'auxiliary': [], 'auxiliary_rate': []},
                 'test': {'loss': [], 'base': [],
                          'auxiliary': [], 'auxiliary_rate': []}}
    print(' start '.center(center, '-'))
    start_point = time.time()
    for epoch in range(epochs):
        epoch_loss = {'train': {'loss': [], 'base': [],
                                'auxiliary': [], 'auxiliary_rate': []},
                      'eval': {'loss': [], 'base': [],
                               'auxiliary': [], 'auxiliary_rate': []},
                      'test': {'loss': [], 'base': [],
                               'auxiliary': [], 'auxiliary_rate': []}}

        cache = None
        for i, pack in enumerate(train):
            inputs = [content.to(device) for content in pack]
            # モデル訓練
            if i == 0:
                pass
            else:
                # キャッシュから１バッチ前のデータで訓練
                loss = training_eval(model, optimizer, criterion, cache,
                                     epoch_loss, 'train')
                # 勾配計算
                loss.backward()
                optimizer.step()

            # モデル評価
            training_eval(model, optimizer, criterion, inputs,
                          epoch_loss, 'eval')
            # データをキャッシュに保存して次回の訓練データにする
            cache = inputs

        # テストデータによる評価
        for pack in test:
            inputs = [content.to(device) for content in pack]
            training_eval(model, optimizer, criterion, inputs,
                          epoch_loss, 'test')

        # 損失データの登録
        appender(loss_pack, epoch_loss, 'train')
        appender(loss_pack, epoch_loss, 'eval')
        appender(loss_pack, epoch_loss, 'test')

        # lossのログを表示
        logger(verbose, epoch, center, epoch_loss)
    print(' complete!! '.center(center, '-'))
    print(f'Execution_time: {round(time.time() - start_point, 3)}')
    return loss_pack


def training_eval(model: Callable[[Tensor], Tensor],
                  optimizer: object,
                  criterion: Callable[[Tensor], Tensor],
                  inputs: Tuple[Tensor],
                  epoch_loss: Dict[str, Dict[str, List[Optional[int]]]],
                  mode: str) -> Optional[int]:
    """モデルへの入力関数"""
    src, tgt, y = inputs
    key = mode
    if mode == 'test':
        mode = 'eval'
    getattr(model, mode)()
    model.train()
    optimizer.zero_grad()
    base, auxiliary, label = model(src, tgt, y)
    base, auxiliary = base.squeeze(2), auxiliary.squeeze(2)
    loss, base_loss, auxiliary_loss = criterion(base, auxiliary, y, label)
    auxiliary_rate = auxiliary / (base + auxiliary)
    epoch_loss[key]['loss'].append(loss.item())
    epoch_loss[key]['base'].append(base_loss.item())
    epoch_loss[key]['auxiliary'].append(auxiliary_loss.item())
    epoch_loss[key]['auxiliary_rate'].append(auxiliary_rate.mean().item())
    if mode == 'train':
        return loss
    return None


def appender(dic1: Dict[str, Dict[str, List[Optional[List[int]]]]],
             dic2: Dict[str, Dict[str, List[Optional[int]]]],
             mode: str) -> None:
    """損失状況の更新"""
    for key, value in dic2[mode].items():
        dic1[mode][key].append(value)
    return None


def logger(verbose: int,
           epoch: int,
           center: int,
           epoch_loss: Dict[str, Dict[str, List[Optional[int]]]]) -> None:
    if verbose == 0:
        return None
    elif epoch % verbose == 0:
        print(f' epoch_{epoch} '.center(center))
        train_mean = torch.mean(
                        torch.tensor(epoch_loss['train']['loss'])
                        ).item()
        valid_mean = torch.mean(
                        torch.tensor(epoch_loss['eval']['loss'])
                        ).item()
        test_mean = torch.mean(
                        torch.tensor(epoch_loss['test']['loss'])
                        ).item()
        auxiliary_rate = torch.mean(
                            torch.tensor(epoch_loss['train']['auxiliary_rate'])
                            ).item()
        print('train_loss: ', round(train_mean, 4),
              '| validation_loss: ', round(valid_mean, 4),
              '| test_loss: ', round(test_mean, 4),
              '| auxiliary_rate: ', round(auxiliary_rate, 4))
        return None


def training(model: Callable[[Tensor], Tensor],
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
