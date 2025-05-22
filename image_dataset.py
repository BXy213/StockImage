import os
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

import config as cfg
from generate_image_data import IMAGE_WIDTH_DICT, IMAGE_HEIGHT_DICT


FREQ_DICT = {5: "week", 20: "month"}


class ImageDataset(Dataset):
    def __init__(self,
        window_size,
        predict_window_size,
        year,
        has_volume_bar=False,
        ma_lag_list=None,
        annual_stocks_num="all",
        tstat_threshold=0,
        regression_label=None,
    ):
        self.ws = window_size
        assert self.ws in [5, 20]
        self.data_freq = FREQ_DICT[self.ws]

        self.pw = predict_window_size
        self.year = year

        self.has_vb = has_volume_bar
        self.ma_lag_list = ma_lag_list

        # Ran: 使用分类变量还是数值变量，见get_label_value()方法
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]

        self.save_dir = os.path.join(cfg.IMAGE_DATA_DIR, f"stock_dataset_all")
        vb_str = "hasvb" if self.has_vb else "novb"
        self.data_file_name = f"{self.ws}d_{self.data_freq}_{vb_str}_{str(self.ma_lag_list)}_ma_{self.year}"
        self.image_save_path = os.path.join(self.save_dir, f"{self.data_file_name}_images.dat")
        self.label_save_path = os.path.join(self.save_dir, f"{self.data_file_name}_labels.feather")

        self.images, self.label_dict = self.load_images_and_labels()

        self.ret_val_name = f"Ret_{FREQ_DICT[self.pw]}"
        self.label = self.get_label_value()

        self.filter_data(annual_stocks_num, tstat_threshold)


    def filter_data(self, annual_stocks_num, tstat_threshold):
        df = pd.DataFrame(
            {
                "StockID": self.label_dict["StockID"],
                "MarketCap": abs(self.label_dict["MarketCap"]),
                "Date": pd.to_datetime([str(t) for t in self.label_dict["Date"]]),
            }
        )
        if annual_stocks_num != "all":
            num_stockid = len(np.unique(df.StockID))
            new_df = df.groupby("StockID").max().copy()
            
            new_df = new_df.sort_values(by=["MarketCap"], ascending=False)
            if len(new_df) > int(annual_stocks_num):
                stockids = new_df.iloc[: int(annual_stocks_num)]["StockID"]
            else:
                stockids = new_df.StockID
            print(
                f"Year {self.year}: select top {annual_stocks_num} stocks ({len(stockids)}/{num_stockid}) stocks for training"
            )
        else:
            stockids = np.unique(df.StockID)
        stockid_idx = pd.Series(df.StockID).isin(stockids)

        idx = (
            stockid_idx
            & pd.Series(self.label != -99)
            & pd.Series(self.label_dict["EWMA_vol"] != 0.0)
        )

        if tstat_threshold != 0:
            tstats = np.divide(
                self.label_dict[self.ret_val_name], np.sqrt(self.label_dict["EWMA_vol"])
            )
            tstats = np.abs(tstats)
            t_th = np.nanpercentile(tstats[idx], tstat_threshold)
            tstat_idx = tstats > t_th
            print(
                f"Before filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
            )
            idx = idx & tstat_idx
            print(
                f"After filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
            )

        self.label = self.label[idx]
        print(f"Year {self.year}: samples size: {len(self.label)}")
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def get_label_value(self):
        print(f"Using {self.ret_val_name} as label")
        ret = self.label_dict[self.ret_val_name]

        print(
            f"Using {self.regression_label} regression label (None represents classification label)"
        )
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)

        return label

    @staticmethod
    def rebuild_image(image, image_name, par_save_dir, image_mode="L"):
        img = Image.fromarray(image, image_mode)
        save_dir = cfg.get_dir(os.path.join(par_save_dir, "images_rebuilt_from_dataset"))
        img.save(os.path.join(save_dir, f"{image_name}.png"))

    @staticmethod
    def load_image_np_data(img_save_path, window_size):
        images = np.memmap(img_save_path, dtype=np.uint8, mode="r")
        images = images.reshape(
            (-1, 1, IMAGE_HEIGHT_DICT[window_size], IMAGE_WIDTH_DICT[window_size])
        )
        return images

    def load_images_and_labels(self):
        print(f"loading images from {self.image_save_path}")
        images = self.load_image_np_data(self.image_save_path, self.ws)

        self.rebuild_image(
            images[0][0],
            image_name=self.data_file_name,
            par_save_dir=self.save_dir,
        )

        label_df = pd.read_feather(self.label_save_path)
        label_df["StockID"] = label_df["StockID"].astype(str)
        label_dict = {c: np.array(label_df[c]) for c in label_df.columns}
        return images, label_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {
            "image": self.images[idx],
            "label": self.label[idx],
            # 此处ret_val为原始数据下一周期Dretwd+1的累乘后减1，计算无误
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx],
        }
        return sample
