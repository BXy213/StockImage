import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw

import get_stock_data as gsd
import config as cfg

windows_size_dict = {"week": 5, "month": 20}

BAR_WIDTH = 3
LINE_WIDTH = 1
IMAGE_WIDTH_DICT = {5: BAR_WIDTH * 5, 20: BAR_WIDTH * 20}
IMAGE_HEIGHT_DICT = {5: 32, 20: 64}

VOLUME_CHART_GAP = 1


class GenerateImageData(object):
    def __init__(self, year, data_freq, ma_lag_list=None, has_volume_bar=False):
        self.year = year
        self.data_freq = data_freq
        assert self.data_freq in ["week", "month"]
        self.ma_lag_list = ma_lag_list
        self.has_volume_bar = has_volume_bar

        self.ret_len_list = [5, 20, 60]
        self.window_size = windows_size_dict[self.data_freq]

        self.image_width, self.image_height = (
            IMAGE_WIDTH_DICT[self.window_size],
            IMAGE_HEIGHT_DICT[self.window_size],
        )

        self.df = None
        self.stock_id_list = None

        self.save_dir = cfg.get_dir(os.path.join(cfg.IMAGE_DATA_DIR, f"stock_dataset_all"))
        self.sample_image_save_dir = cfg.get_dir(os.path.join(cfg.IMAGE_DATA_DIR, "sample_images"))

        vb_str = "hasvb" if self.has_volume_bar else "novb"
        self.file_name = f"{self.window_size}d_{self.data_freq}_{vb_str}_{str(self.ma_lag_list)}_ma_{self.year}"
        self.log_file_name = os.path.join(self.save_dir, f"{self.file_name}.txt")
        self.labels_filename = os.path.join(self.save_dir, f"{self.file_name}_labels.feather")
        self.images_filename = os.path.join(self.save_dir, f"{self.file_name}_images.dat")

    def get_feature_and_dtype_list(self):
        float32_features = (
            [
                "EWMA_vol",
                "Ret",
                "Ret_week",
                "Ret_month",
                "MarketCap",
            ]
            + [f"Ret_{i}d" for i in self.ret_len_list]
        )
        uint8_features = ["image", "window_size"]
        object_features = ["StockID"]
        datetime_features = ["Date"]

        feature_list = float32_features + uint8_features + object_features + datetime_features

        float32_dict = {feature: np.float32 for feature in float32_features}
        uint8_dict = {feature: np.uint8 for feature in uint8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        dtype_dict = {
            **float32_dict,
            **uint8_dict,
            **object_dict,
            **datetime_dict,
        }
        return dtype_dict, feature_list

    @staticmethod
    def adjust_price(df):
        '''
        根据前一个交易日的收盘价和当日的Return，计算调整后的相对价格
        '''
        if len(df) == 0:
            raise Exception("adjust_price: Empty Dataframe")
        if len(df.Date.unique()) != len(df):
            raise Exception("adjust_price: Dates not unique")
        df = df.reset_index(drop=True)

        fd_close = abs(df.at[0, "Close"])
        if df.at[0, "Close"] == 0.0 or pd.isna(df.at[0, "Close"]):
            raise Exception("adjust_price: First day close is nan or zero")

        pre_close = fd_close
        res_df = df.copy()

        res_df.at[0, "Close"] = 1.0
        res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / pre_close
        res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / pre_close
        res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / pre_close

        pre_close = 1
        for i in range(1, len(res_df)):
            today_closep = abs(res_df.at[i, "Close"])
            today_openp = abs(res_df.at[i, "Open"])
            today_highp = abs(res_df.at[i, "High"])
            today_lowp = abs(res_df.at[i, "Low"])
            today_ret = np.float64(res_df.at[i, "Ret"])

            res_df.at[i, "Close"] = (1 + today_ret) * pre_close
            res_df.at[i, "Open"] = res_df.at[i, "Close"] / today_closep * today_openp
            res_df.at[i, "High"] = res_df.at[i, "Close"] / today_closep * today_highp
            res_df.at[i, "Low"] = res_df.at[i, "Close"] / today_closep * today_lowp

            if not pd.isna(res_df.at[i, "Close"]):
                pre_close = res_df.at[i, "Close"]
            else:
                raise Exception("adjust_price: close is nan")

        return res_df

    def load_adjusted_daily_prices(self, stock_df, date):
        '''
        获取date开始往前，self.window_size长度的调整后的股票价格数据，
        股票价格数据以date日Close为基准，取相对价格。
        ma（移动平均）也为相对价格的均值。
        '''
        if date not in set(stock_df.Date):
            return 0
        date_index = stock_df[stock_df.Date == date].index[0]
        ma_offset = 0 if self.ma_lag_list is None else np.max(self.ma_lag_list)
        data = stock_df.loc[(date_index - (self.window_size - 1) - ma_offset) : date_index]
        if len(data) < self.window_size:
            return 1

        # 看下数据量够不够，不够就不画移动平均线了
        if len(data) < (self.window_size + ma_offset):
            ma_lag_list = []
            data = stock_df.loc[(date_index - (self.window_size - 1)) : date_index]
        else:
            ma_lag_list = self.ma_lag_list

        data = self.adjust_price(data)
        start_date_index = data.index[-1] - self.window_size + 1
        data[["Open", "High", "Low", "Close"]] *= 1.0 / data["Close"].loc[start_date_index]

        if self.ma_lag_list is not None:
            for i, ma_lag in enumerate(ma_lag_list):
                ma_name = "ma" + str(ma_lag)
                data[ma_name] = data["Close"].rolling(ma_lag).mean()
        data["Prev_Close"] = data["Close"].shift(1)

        df = data.loc[start_date_index:]
        if (len(df) != self.window_size or np.around(df.iloc[0]["Close"], decimals=3) != 1.000):
            raise Exception("error in load_adjusted_daily_prices")
        df = df.reset_index(drop=True)
        return df.copy(), ma_lag_list

    def generate_daily_features(self, stock_df, date):
        '''
        根据load_adjusted_daily_prices()的数据生成图片，并返回全部所需的特征数据。
        错误返回：
        0: date不在stock_id股票数据中(正好停牌等)
        1: 数据长度不够
        2: 无法绘制图像
        '''
        res = self.load_adjusted_daily_prices(stock_df, date)
        if isinstance(res, int):
            return res
        df, ma_lag_list = res
        image_generator_obj = ImageGenerator(
            df=df, has_volume_bar=self.has_volume_bar, ma_lag_list=ma_lag_list)
        image_data = image_generator_obj.draw_image()
        if image_data is None:
            return 2

        last_day = df[df.Date == date].iloc[0]
        feature_dict = {feature: last_day[feature] for feature in stock_df.columns}
        
        feature_dict["image"] = image_data
        feature_dict["window_size"] = self.window_size
        feature_dict["Date"] = date
        return feature_dict

    def generate_image_data(self, for_test=False):
        '''
        按年份生成训练所需的各类数据
        images.dat按顺序保存图片数据
        labels.feather按顺序保存标签数据
        log.txt统计观测值个数和缺失值个数
        '''
        if (os.path.isfile(self.log_file_name)
            and os.path.isfile(self.labels_filename)
            and os.path.isfile(self.images_filename)):
            print("Found pregenerated file {}".format(self.file_name))
            return

        print(f"Generating {self.file_name}")

        df = gsd.get_processed_data_by_year(self.year)
        stock_id_list = np.unique(df.index.get_level_values("StockID"))

        if for_test:
            stock_id_list = stock_id_list[:10]
            # print("---")
            # print(stock_id_list)
            # print("---")

        dtype_dict, feature_list = self.get_feature_and_dtype_list()
        data_miss = [0] * 3
        data_dict = {
            feature: np.empty(len(stock_id_list) * 60, dtype=dtype_dict[feature])
            for feature in feature_list
        }

        data_dict["image"] = np.empty(
            [len(stock_id_list) * 60, self.image_width * self.image_height],
            dtype=dtype_dict["image"],
        )
        data_dict["image"].fill(0)

        sample_num = 0
        for i, stock_id in enumerate(tqdm(stock_id_list)):
            stock_df = df.xs(stock_id, level=1).copy()
            stock_df = stock_df.reset_index()
            dates = gsd.get_freq_end_dates(self.data_freq)
            dates = dates[dates.year == self.year]
            for j, date in enumerate(dates):
                # stock_id X date 数据
                # print(stock_id)
                # print(date)
                daily_data = self.generate_daily_features(stock_df, date)
                if isinstance(daily_data, dict):
                    if (i < 2) and (j == 0):
                        daily_data["image"].save(
                            os.path.join(self.sample_image_save_dir, 
                                         f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png"))

                    daily_data["StockID"] = stock_id
                    im_arr = np.frombuffer(daily_data["image"].tobytes(), dtype=np.uint8)
                    assert im_arr.size == self.image_width * self.image_height

                    data_dict["image"][sample_num, :] = im_arr[:]
                    for feature in [x for x in feature_list if x != "image"]:
                        data_dict[feature][sample_num] = daily_data[feature]
                    sample_num += 1
                elif isinstance(daily_data, int):
                    data_miss[daily_data] += 1
                    # print(f"type: {daily_data}, missed stock: {stock_id}, date: {date}")
                else:
                    raise ValueError

        for feature in feature_list:
            data_dict[feature] = data_dict[feature][:sample_num]

        fp_x = np.memmap(
            self.images_filename,
            dtype=np.uint8,
            mode="w+",
            shape=data_dict["image"].shape,
        )
        fp_x[:] = data_dict["image"][:]
        del fp_x
        print(f"Save image data to {self.images_filename}")

        data_dict = {k: data_dict[k] for k in data_dict.keys() if k != "image"}
        pd.DataFrame(data_dict).to_feather(self.labels_filename)
        print(f"Save label data to {self.labels_filename}")

        with open(self.log_file_name, "w+") as log_file:
            log_file.write(f"total_dates:{sample_num} total_missing:{sum(data_miss)} "
                           f"type0:{data_miss[0]} type1:{data_miss[1]} type2:{data_miss[2]}")
        print(f"Save log file to {self.log_file_name}")


class ImageGenerator(object):
    def __init__(self, df, has_volume_bar=False, ma_lag_list=None):
        self.has_volume_bar = has_volume_bar
        self.volumes = df["Volume"].abs() if has_volume_bar else None

        self.ma_lag_list = ma_lag_list
        if self.ma_lag_list is not None:
            self.ma_name_list = ["ma" + str(ma_lag) for ma_lag in self.ma_lag_list]
        else:
            self.ma_name_list = []
        self.df = df[["Open", "High", "Low", "Close"] + self.ma_name_list].abs()

        self.window_size = len(df)
        assert self.window_size in [5, 20]
        self.min_price = self.df.min().min()
        self.max_price = self.df.max().max()

        self.image_width, self.image_height = (
            IMAGE_WIDTH_DICT[self.window_size],
            IMAGE_HEIGHT_DICT[self.window_size],
        )
        if self.has_volume_bar:
            self.volume_height = int(self.image_height / 5)
            self.image_height -= self.volume_height + VOLUME_CHART_GAP
        else:
            self.volume_height = 0
        
        first_center = (BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + BAR_WIDTH * self.window_size,
            BAR_WIDTH,
            dtype=int
        )

    def __ret_to_yaxis(self, ret):
        pixels_per_unit = (self.image_height - 1.0) / (self.max_price - self.min_price)
        res = np.around((ret - self.min_price) * pixels_per_unit)
        return int(res)

    def draw_candlestick(self):
        candlestick_chart = Image.new(
            mode="L", size=(self.image_width, self.image_height), color=0)
        
        # 绘制移动平均线
        for ma_name in self.ma_name_list:
            ma_data = self.df[ma_name]
            draw = ImageDraw.Draw(candlestick_chart)
            for i in range(self.window_size - 1):
                if np.isnan(ma_data[i]) or np.isnan(ma_data[i + 1]):
                    continue
                draw.line(
                    (self.centers[i], self.__ret_to_yaxis(ma_data[i]),
                     self.centers[i + 1], self.__ret_to_yaxis(ma_data[i + 1]) ),
                    width=1,
                    fill=255,
                )
        
        # 绘制蜡烛图主体部分
        pixels = candlestick_chart.load()
        for i in range(self.window_size):
            highp_today = self.df["High"].iloc[i]
            lowp_today = self.df["Low"].iloc[i]
            closep_today = self.df["Close"].iloc[i]
            openp_today = self.df["Open"].iloc[i]

            if np.isnan(highp_today) or np.isnan(lowp_today) or \
                np.isnan(openp_today) or np.isnan(closep_today):
                continue

            left = int(math.ceil(self.centers[i] - int(BAR_WIDTH / 2)))
            right = int(math.floor(self.centers[i] + int(BAR_WIDTH / 2)))

            line_left = int(math.ceil(self.centers[i] - int(LINE_WIDTH / 2)))
            line_right = int(math.floor(self.centers[i] + int(LINE_WIDTH / 2)))

            line_bottom = self.__ret_to_yaxis(lowp_today)
            line_up = self.__ret_to_yaxis(highp_today)

            # 中线
            for x in range(line_left, line_right + 1):
                for y in range(line_bottom, line_up + 1):
                    pixels[x, y] = 255

            # 左线开盘价
            open_line = self.__ret_to_yaxis(openp_today)
            for x in range(left, int(self.centers[i]) + 1):
                y = open_line
                pixels[x, y] = 255

            # 右线收盘价
            close_line = self.__ret_to_yaxis(closep_today)
            for x in range(int(self.centers[i]) + 1, right + 1):
                y = close_line
                pixels[x, y] = 255

        return candlestick_chart

    def draw_volume_bar(self):
        volume_bar = Image.new(
            mode="L", size=(self.image_width, self.volume_height), color=0)
        max_volume = np.max(self.volumes)

        if (not np.isnan(max_volume)) and max_volume != 0:
            pixels_per_volume = 1.0 * self.volume_height / max_volume
            if not np.around(pixels_per_volume * max_volume) == self.volume_height:
                raise Exception
            draw = ImageDraw.Draw(volume_bar)
            for i in range(self.window_size):
                if np.isnan(self.volumes[i]):
                    continue
                vol_height = int(np.around(self.volumes[i] * pixels_per_volume))
                draw.line(
                    (self.centers[i], 0, self.centers[i], vol_height - 1),
                    width=1,
                    fill=255,
                )
        return volume_bar

    def draw_image(self):
        '''
        生成所需图像，通过draw_candlestick()绘制上方蜡烛图主体部分，通过和draw_volume_bar()绘制下方成交量柱状图。
        无法绘制的图像返回None。
        '''
        if self.max_price == self.min_price or pd.isna(self.max_price) or pd.isna(self.min_price):
            return None
        if not(self.__ret_to_yaxis(self.min_price) == 0 and
            self.__ret_to_yaxis(self.max_price) == self.image_height - 1):
            return None

        candlestick_chart = self.draw_candlestick()

        if self.has_volume_bar:
            volume_bar = self.draw_volume_bar()
            image = Image.new(
                mode="L",
                size=(
                    self.image_width,
                    self.image_height + self.volume_height + VOLUME_CHART_GAP,
                ),
            )
            image.paste(candlestick_chart, (0, self.volume_height + VOLUME_CHART_GAP))
            image.paste(volume_bar, (0, 0))
        else:
            image = candlestick_chart

        # 上下翻转图像
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
