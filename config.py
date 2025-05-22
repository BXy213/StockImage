import os
import pandas as pd


def get_dir(dir: str) -> str:
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir


def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def print_device_profile():
    from multiprocessing import cpu_count
    print(cpu_count())
    import torch
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


def rank_corr(df, col1, col2, method="spearman"):
    if method == "spearman":
        col1_series = df[col1].rank(method="average", ascending=False)
        col2_series = df[col2].rank(method="average", ascending=False)
    else:
        col1_series = df[col1]
        col2_series = df[col2]
    return col2_series.corr(col1_series, method=method)


TEMP_TEST_DIR = r"D:\Users\vbxy2\Desktop"


RAW_DATA_DIR = r"E:\pcp\StockPredictionCNN\CSMAR\日个股回报率文件all"
MARKET_DATA_DIR = r"E:\pcp\StockPredictionCNN\CSMAR\市场回报率文件"

IMAGE_DATA_DIR = r"./SAVED_IMAGE_DATA"

WORK_DIR = r"./WORK_SPACE"
