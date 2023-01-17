import pandas as pd
import numpy as np

class F_Engineering():

    def __init__(self, train, test):
        self.train = train
        self.test = test

    # 店舗ごとの販売合計数を集計
    def _shop_total(self):

        shop_total = self.train.groupby("shop_id").sum()
        shop_total = shop_total.drop(['date_block_num','item_id', 'item_price'], axis = 1)
        shop_total.rename(columns = {"item_cnt_day" : "item_cnt_total"}, inplace = True)

        return shop_total
    
    # 店舗ごとの重みを計算
    def _culc_w_shop(self):
        shop_total = self._shop_total()
        
        ## 販売数合計
        item_total = np.sum(shop_total.values)

        ## 重みのリスト(shop数)
        w = np.zeros(shop_total.shape[0])

         ## 重み　＝　各店舗 ÷ 販売数合計 × 100
        for i in range(shop_total.shape[0]):
            w[i] = shop_total.iloc[i].values / item_total * 100

        shop_total["W"] = w
        
        return shop_total

    def output_w(self):
        
        shop_total = self._culc_w_shop()

        # indexである[shop_id]をcolumnに追加
        shop_total["shop_id"] = shop_total.index

        # for文で使用するリストを作成 --> jに用いる
        inlsit = list(shop_total.index)

        # df_testに重み用の空の箱を作る
        w_test = np.zeros(len(self.test))

        # テストデータが持つショップIDと合致するショップIDの重みをdf_test["W"]に格納する。
        for i in range(self.test.shape[0]):
            for j in inlsit:
                if self.test["shop_id"][i] != shop_total["shop_id"][j]:
                    pass
                else:
                    w_test[i] = shop_total["W"][j]

        # [W]列を作成し、重みを加える
        self.test["W"] = w_test
        # df_test["W"]の合計値を計算
        w_total = np.sum(self.test.W)

        # df_test["W"]の合計値を元に、それぞれの重みを更新。 --> 全体で１になるように。
        for i in range(self.test.shape[0]):
            self.test["W"][i] /= w_total

        return self.test
