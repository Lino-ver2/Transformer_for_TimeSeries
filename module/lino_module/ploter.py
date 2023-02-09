import datetime

import pandas as pd
import torch
import matplotlib.pyplot as plt


def learning_plot(train_loss,
                  validation_loss,
                  test_loss,
                  img_path,
                  name,
                  scaler,
                  figsize,
                  saving=True):
    """訓練データから学習曲線をプロットする"""
    plt.figure(figsize=figsize)
    test_loss_list = [torch.mean(i) for i in test_loss]
    plt.plot([torch.mean(i) for i in train_loss], label=('train_loss'))
    plt.plot([torch.mean(i) for i in validation_loss], label='validation_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.legend()
    if scaler.__name__ == 'MinMaxScaler':
        plt.yticks([round(i*1e-2, 2) for i in range(1, 10)])
        plt.ylim(0, 0.1)
    if scaler.__name__ == 'StandardScaler':
        plt.yticks([round(i*1e-1, 2) for i in range(1, 10)])
        plt.ylim(0, 1)

    plt.grid(axis='x')
    plt.grid(axis='y')
    loss_title = round(test_loss_list[-1].item(), 4)
    plt.title(f'Test Loss: {loss_title}\n' + name)
    img_path = img_path
    loss_name = f'Loss({name}).png'
    if saving:
        plt.savefig(img_path + loss_name)
    plt.show()
    return None


def confirmation(model, train, test, device):
    """訓練データとテストデータを使った推測(教師強制と同じ推論であることに注意)"""
    model.eval()
    train_preds = []
    model_name = model._get_name()
    for src, tgt, _ in train:
        train_pred = model(src.to(device), tgt.to(device))
        if model_name == 'WithAuxiliary':
            train_pred = train_pred[0] + train_pred[1]
        train_preds.append(train_pred[:, -1].cpu())

    test_preds = []
    for src, tgt, _ in test:
        test_pred = model(src.to(device), tgt.to(device))
        if model_name == 'WithAuxiliary':
            test_pred = test_pred[0] + test_pred[1]
        test_preds.append(test_pred[:, -1].cpu())
    return train_preds, test_preds


def val_time_series(original,
                    train_preds,
                    test_preds,
                    scaler,
                    d_model,
                    dilation,
                    seq):
    """推測値をプロットできるようにインデックスを整える"""
    # 比較用に訓練に使用した時系列データを用意
    # 訓練データ, テストデータとのラグを計算
    lag = d_model * (dilation + 1) + seq

    fit_target = original.values.reshape(-1, 1)
    src = scaler().fit(fit_target)

    # 予測データを ndarray に変換してプロットできるようにする
    train_pred = torch.concat(train_preds).reshape(-1).detach().numpy()
    test_pred = torch.concat(test_preds).reshape(-1).detach().numpy()

    # 予測データの標準化を解除
    train_pred = src.inverse_transform(train_pred.reshape(-1, 1)).reshape(-1)
    test_pred = src.inverse_transform(test_pred.reshape(-1, 1)).reshape(-1)

    # 訓練データラベルのラグを修正
    tr_start = original.index[0] + datetime.timedelta(lag)
    tr_end = tr_start + datetime.timedelta(len(train_pred) - 1)
    tr_idx = pd.date_range(tr_start, tr_end)
    # ラグを修正したインデックスでプロット用の訓練予測データを作成
    train_time_series = pd.Series(train_pred, index=tr_idx)

    # テストデータのラグを修正
    te_start = tr_end + datetime.timedelta(1)
    te_end = te_start + datetime.timedelta(len(test_pred) - 1)
    te_idx = pd.date_range(te_start, te_end)
    # ラグを修正したインデックスでプロロット用のテスト予測データを作成
    test_time_series = pd.Series(test_pred, index=te_idx)
    return train_time_series, test_time_series, original


def uni_time_series(original,
                    train_preds,
                    test_preds,
                    scaler,
                    d_model,
                    dilation,
                    seq):
    """推測値をプロットできるようにインデックスを整える"""
    # 比較用に訓練に使用した時系列データを用意
    # 訓練データ, テストデータとのラグを計算
    lag = seq * (dilation + 1)

    fit_target = original.values.reshape(-1, 1)
    src = scaler().fit(fit_target)

    # 予測データを ndarray に変換してプロットできるようにする
    train_pred = torch.concat(train_preds).reshape(-1).detach().numpy()
    test_pred = torch.concat(test_preds).reshape(-1).detach().numpy()

    # 予測データの標準化を解除
    train_pred = src.inverse_transform(train_pred.reshape(-1, 1)).reshape(-1)
    test_pred = src.inverse_transform(test_pred.reshape(-1, 1)).reshape(-1)

    # 訓練データラベルのラグを修正
    tr_start = original.index[0] + datetime.timedelta(lag)
    tr_end = tr_start + datetime.timedelta(len(train_pred) - 1)
    tr_idx = pd.date_range(tr_start, tr_end)
    # ラグを修正したインデックスでプロット用の訓練予測データを作成
    train_time_series = pd.Series(train_pred, index=tr_idx)

    # テストデータのラグを修正
    te_start = tr_end + datetime.timedelta(1)
    te_end = te_start + datetime.timedelta(len(test_pred) - 1)
    te_idx = pd.date_range(te_start, te_end)
    # ラグを修正したインデックスでプロロット用のテスト予測データを作成
    test_time_series = pd.Series(test_pred, index=te_idx)
    return train_time_series, test_time_series, original


def confirmation_plot(train_time_series,
                      test_time_series,
                      original, img_path,
                      name,
                      figsize,
                      saving=True):
    """推測値のプロット"""
    plt.figure(figsize=figsize)
    plt.plot(original, alpha=0.5, label='true')
    plt.plot(train_time_series, label='train_preds')
    plt.plot(test_time_series, label='test_preds')
    plt.grid(axis='x')
    plt.title(name)
    plt.legend()

    predict_name = f'Predict({name}).png'
    if saving:
        plt.savefig(img_path + predict_name)
    plt.show()
    return None


# 再帰推論用プロット
def inference_ploter(ori, pred, label, title, path, figsize, saving):
    plt.figure(figsize=figsize)
    plt.plot(ori, label='origin', alpha=0.5)
    plt.plot(pred, label=label)
    plt.grid(axis='x')
    plt.title(title)
    plt.legend()
    if saving:
        plt.savefig(path)
    plt.show()
    return None
