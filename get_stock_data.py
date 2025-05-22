import os
import numpy as np
import pandas as pd

import config as cfg


def get_processed_data():
    '''
    读取处理后的CSMAR数据，以DataFrame格式返回。
    '''
    stock_daily_data_path = os.path.join(cfg.RAW_DATA_DIR, "stock_daily.parquet")
    
    if os.path.exists(stock_daily_data_path):
        print(f"Loading data from pregenerated data file: {stock_daily_data_path}")
        df = pd.read_parquet(stock_daily_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace = True)
        return df.copy()

    raw_df = get_raw_stock_data()
    processed_df = process_raw_data(raw_df)
    processed_df.reset_index().to_parquet(stock_daily_data_path, index = False)
    return processed_df.copy()
    

def get_raw_stock_data():
    '''
    读取CSMAR中原始数据，以DataFrame格式返回。
    '''
    raw_df_list = []
    for filename in os.listdir(cfg.RAW_DATA_DIR):
        if not filename.endswith('.csv'):
            continue
        data_path = os.path.join(cfg.RAW_DATA_DIR, filename)
    
        if os.path.exists(data_path):
            print(f"Loading raw data from: {data_path}")
            df = pd.read_csv(data_path, header = 0, parse_dates = ['Trddt'])
            df = df.loc[(df['Markettype'] == 1) | (df['Markettype'] == 4)]
            df = df[['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnvaltrd', 'Dsmvosd', 'Dretwd']]
            df = df.rename(
                columns={
                    "Trddt": "Date",
                    "Stkcd": "StockID",
                    "Loprc": "Low",
                    "Hiprc": "High",
                    "Clsprc": "Close",
                    "Dnvaltrd": "Volume", #日个股交易金额
                    "Dsmvosd": "MarketCap", #日个股流通市值
                    "Opnprc": "Open",
                    "Dretwd": "Ret", #考虑现金红利再投资的日个股回报率，值为0.01，代表上涨1%
                }
            )
            raw_df_list.append(df)
        else:
            raise Exception(f"File {data_path} not exists")

    raw_data_df = pd.concat(raw_df_list)
    raw_data_df = raw_data_df.reset_index(drop = True)
    return raw_data_df.copy()


def process_raw_data(df):
    '''
    清洗并处理CSMAR原始数据。
    '''
    df.StockID = df.StockID.astype(str)
    df.Ret = df.Ret.astype(np.float64)
    df = df.replace(
        {
            "Close": {0.0: np.nan},
            "Open": {0.0: np.nan},
            "High": {0.0: np.nan},
            "Low": {0.0: np.nan},
            "Volume": {0.0: np.nan},
        }
    )
    df = df.dropna(subset=["Ret", "Volume", "Close", "Open", "High", "Low", "MarketCap"])
    df[["Close", "Open", "High", "Low", "Volume", "MarketCap"]] = df[
        ["Close", "Open", "High", "Low", "Volume", "MarketCap"]
    ].abs()
    df.set_index(["Date", "StockID"], inplace=True)

    df = df[~df.index.duplicated()]

    df.sort_index(inplace=True)
    df["log_ret"] = np.log(1 + df.Ret)
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )

    for freq in ["week", "month"]:
        period_end_dates = get_freq_end_dates(freq)
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        freq_df["freq_ret"] = freq_df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1 # 计算的是往后一个周期的回报率，shift(-1)向上平移一行
        )
        df[f"Ret_{freq}"] = freq_df["freq_ret"]
        
    for i in [5, 20, 60]:
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1 #计算的是未来5/20/60日的回报率
        )
    return df


def get_freq_end_dates(freq):
    assert freq in ["week", "month"]
    if freq == "week":
        df = get_trading_dates()
        df["ISOWeek"] = df["Date"].dt.isocalendar().week
        df["ISOYear"] = df["Date"].dt.isocalendar().year
        df_week = df.groupby(["ISOYear", "ISOWeek"])["Date"].max().to_frame()
        df_week.set_index(["Date"], inplace=True)
        return df_week.index
    elif freq == "month":
        df = get_trading_dates()
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df_month = df.groupby(["Year", "Month"])["Date"].max().to_frame()
        df_month.set_index(["Date"], inplace=True)
        return df_month.index
    else:
        raise Exception("Unknown freq")


def get_trading_dates():
    df = pd.read_csv(
        os.path.join(cfg.MARKET_DATA_DIR, "TRD_Cndalym.csv"),
        header = 0,
        parse_dates=["Trddt"],
    )
    df = df.loc[(df['Markettype'] == 5)]
    df = df[['Trddt']]
    df = df.rename(
        columns={
            "Trddt": "Date",
        }
    )
    df.set_index(["Date"], inplace=True)
    df.sort_index(inplace = True)
    df.reset_index(inplace = True)
    return df


def get_processed_data_by_year(year):
    '''
    返回的DataFrame包含：

    1. 索引 multi index:
    (1) Date: 日期, "datetime64[ns]", level=0
    (2) StockID: 股票代码, str, level=1

    2. 列columns: 均为float
    (1) Open: 开盘价 
    (2) High: 最高价
    (3) Low: 最低价
    (4) Close: 收盘价
    (5) Volume: 日个股交易金额
    (6) MarketCap: 日个股流通市值，计算公式为个股的流通股数与收盘价的乘积Shares * Close

    (7) Ret: 日个股回报率 (值为0.01则代表上涨1%)
    (8) log_ret: 日个股回报率的对数
    (9) cum_log_ret: 日个股回报率累计值
    (10) EWMA_vol: Ret的指数加权移动平均波动率

    (11) Ret_week: 未来一周的回报率，仅在week_end_dates即最后一天有观测值
    (12) Ret_month: 未来一个月的回报率，仅在month_end_dates即最后一天有观测值
    (13) Ret_5d: 未来5个交易日的回报率
    (14) Ret_20d: 未来20个交易日的回报率
    (15) Ret_60d: 未来60个交易日的回报率
    '''
    df = get_processed_data()
    df = df[
        df.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
    ].copy()
    return df


def get_mkt_freq_rets(freq):
    assert freq in ["week", "month"]
    df = pd.read_csv(
        os.path.join(cfg.MARKET_DATA_DIR, "TRD_Cndalym.csv"),
        header = 0,
        parse_dates=["Trddt"],
    )
    df = df.rename(
        columns={
            "Trddt": "Date",
            "Cdretwdeq": "ewret",
            "Cdretwdos": "vwret"
        }
    )
    df = df.loc[(df['Markettype'] == 5)]
    df = df[["Date", "ewret", "vwret"]]
        
    df["ewret"] = df.ewret.astype(np.float64)
    df["vwret"] = df.vwret.astype(np.float64)
    df = df.dropna(subset = ["ewret", "vwret"])
    df = df.set_index("Date")
    df.sort_index(inplace = True)
    
    df["log_ewret"] = np.log(1 + df.ewret)
    df["log_vwret"] = np.log(1 + df.vwret)
    df["cum_log_ewret"] = df["log_ewret"].cumsum(skipna = True)
    df["cum_log_vwret"] = df["log_vwret"].cumsum(skipna = True)
    
    period_end_dates = get_freq_end_dates(freq)
    freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
    
    # 计算的是往后一个周期的回报率(不包含当日)
    freq_df["next_freq_ewret"] = np.exp(freq_df["cum_log_ewret"].shift(-1) - freq_df["cum_log_ewret"]) - 1
    freq_df["next_freq_vwret"] = np.exp(freq_df["cum_log_vwret"].shift(-1) - freq_df["cum_log_vwret"]) - 1

    freq_df = freq_df[["next_freq_ewret", "next_freq_vwret"]]
    freq_df = freq_df.dropna(subset = ["next_freq_ewret", "next_freq_vwret"])

    # freq_df.reset_index().to_csv(os.path.join(dcf.MARKET_DATA_DIR, "tmp.csv"), index = False)
    return freq_df
