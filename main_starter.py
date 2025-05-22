from ast import Set
import pandas as pd
import numpy as np


def try_num_worker():
    from time import time
    import multiprocessing as mp
    import torch
    import torchvision
    from torchvision import transforms
 
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
 
    trainset = torchvision.datasets.MNIST(
        root='dataset/',
        train=True,  #如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
        download=True, #如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
        transform=transform
    )
 
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


def trytry():
    import get_stock_data as gsd
    year = 2019
    df = gsd.get_processed_data_by_year(year)
    stock_id_list = np.unique(df.index.get_level_values("StockID"))
    stock_id_list = ['156']

    for i, stock_id in enumerate(stock_id_list):
        stock_df = df.xs(stock_id, level="StockID").copy()
        stock_df = stock_df.reset_index()
        # dates = gsd.get_period_end_dates("week")
        dates = stock_df.Date
        dates = dates[dates.dt.year == year]
        dates = dates[dates.dt.month <= 11]
        dates = dates[dates.dt.month >= 9]
        print(dates.values)
        # for j, date in enumerate(dates):
        print(stock_df.loc[(stock_df["Date"].isin(dates.values)), ["Date", "Open"]])


def try1():
    import os
    from generate_image_data import GenerateImageData
    
    for year in range(1991, 2024, 1):
        gd = GenerateImageData(year=year, data_freq="week", ma_lag_list=[20], has_volume_bar=True)
        gd.generate_image_data(for_test=False)
        del gd

    for year in range(1991, 2024, 1):
        gd = GenerateImageData(year=year, data_freq="month", ma_lag_list=[20], has_volume_bar=True)
        gd.generate_image_data(for_test=False)
        del gd




def try2():
    import os
    from PIL import Image
    from image_dataset import ImageDataset
    import config as cfg

    tmp_dataset = ImageDataset(window_size=5, predict_window_size=5, year=2019, has_volume_bar=True,
                               ma_lag_list=[20])
    # tmp_dataset[0]['image']
    img = Image.fromarray(tmp_dataset[0]['image'][0], "L")
    img.save(os.path.join(cfg.TEMP_TEST_DIR, f"aaa.png"))
    print(tmp_dataset[0])

def try3():
    from solver import Set_model
    # Set_model(window_size=5, predict_window_size=5)
    Set_model(window_size=20, predict_window_size=20)


def try4():
    import torch
    from model import ModelManager
    model_manager = ModelManager(window_size=5)
    mdl = model_manager.init_model()
    model_manager.model_summary(output_path=r'D:\Users\vbxy2\Desktop\t.txt')


if __name__ == "__main__":
    # r"D:\Users\vbxy2\Desktop\a.csv"
    # trytry()
    # try1()
    # try2()
    # try3()
    # try4()
    pass
