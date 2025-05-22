import time
import os
import pickle
import copy
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset, random_split

import config as cfg
import get_stock_data as gsd
from image_dataset import ImageDataset
from model import ModelManager
from portfolio import PortfolioManager

FREQ_DICT = {5: "week", 20: "month"}
IN_SAMPLE_YEARS = list(range(2007, 2017))
OUT_OF_SAMPLE_YEARS = list(range(2017, 2024))

BATCH_SIZE = 128
NUM_WORKERS = 1


class Experiment(object):
    def __init__(
        self,
        window_size,
        predict_window_size,

        ma_lag_list=None,
        has_volume_bar=False,

        ensemble_nums=5,
        learning_rate=1e-5,
        drop_prob=0.50,
        device_number=0,
        max_epoch=50,
        early_stop=True,
        
        is_year_list=IN_SAMPLE_YEARS,
        oos_year_list=OUT_OF_SAMPLE_YEARS,

        annual_stocks_num="all",
        tstat_threshold=0,
        regression_label=None,

        weight_decay=0,
        loss_name="cross_entropy",
        train_size_ratio=0.7,
    ):
        self.ws = window_size
        self.pw = predict_window_size
        self.pf_freq = FREQ_DICT[self.pw]

        self.ensem = ensemble_nums
        self.lr = learning_rate
        self.drop_prob = drop_prob

        self.device_number = device_number if device_number is not None else 0
        self.device = torch.device(
            "cuda:{}".format(self.device_number) if torch.cuda.is_available() else "cpu"
        )
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.ma_lag_list = ma_lag_list
        self.has_volume_bar = has_volume_bar

        self.is_year_list = is_year_list
        self.oos_year_list = oos_year_list

        self.annual_stocks_num = annual_stocks_num
        self.tstat_threshold = tstat_threshold
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]
        self.label_dtype = torch.long if self.regression_label is None else torch.float

        self.model_obj = ModelManager(window_size=self.ws, drop_prob=self.drop_prob, regression_label=self.regression_label)
        self.weight_decay = weight_decay
        self.loss_name = loss_name if self.regression_label is None else "MSE"

        self.train_size_ratio = train_size_ratio    

        model_name = self.model_obj.name

        self.exp_name = self.get_exp_name()
        self.exp_dir = cfg.get_dir(os.path.join(cfg.WORK_DIR, self.exp_name))
        self.model_dir = cfg.get_dir(os.path.join(self.exp_dir, f"model-{model_name}"))
        self.model_obj.model_summary(output_path=os.path.join(self.model_dir, "model_struct.txt"))

        self.ensem_res_dir = cfg.get_dir(os.path.join(self.model_dir, "ensem_res"))
        self.oos_metrics_path = os.path.join(self.ensem_res_dir, "oos_metrics.pkl")
        self.portfolio_dir = cfg.get_dir(os.path.join(self.exp_dir, f"portfolio-{model_name}"))

    def get_exp_name(self):
        exp_setting_list = [
            f"exp-{self.ws}d{self.pw}p-lr{self.lr:.0E}-dp{self.drop_prob:.2f}",
            f"ma{str(self.ma_lag_list)}-vb{self.has_volume_bar}",
        ]
        if self.loss_name != "cross_entropy":
            exp_setting_list.append(self.loss_name)
        if self.annual_stocks_num != "all":
            exp_setting_list.append(f"top{self.annual_stocks_num}AnnualStock")
        if self.tstat_threshold != 0:
            exp_setting_list.append(f"{self.tstat_threshold}tstat")
        if self.train_size_ratio != 0.7:
            exp_setting_list.append(f"tv_ratio{self.train_size_ratio:.1f}")
        if self.model_obj.regression_label is not None:
            exp_setting_list.append(self.model_obj.regression_label)
        exp_name = "-".join(exp_setting_list)
        return exp_name

    def get_train_validate_dataloaders_dict(self):
        tv_datasets = {
            year: ImageDataset(
                window_size=self.ws,
                predict_window_size=self.pw,
                year=year,
                has_volume_bar=self.has_volume_bar,
                ma_lag_list=self.ma_lag_list,
                annual_stocks_num=self.annual_stocks_num,
                tstat_threshold=self.tstat_threshold,
                regression_label=self.model_obj.regression_label
            )
            for year in self.is_year_list
        }
        tv_dataset = ConcatDataset([tv_datasets[year] for year in self.is_year_list])
        train_size = int(len(tv_dataset) * self.train_size_ratio)
        validate_size = len(tv_dataset) - train_size
        print(
            f"Training and validation data from {self.is_year_list[0]} to {self.is_year_list[-1]} \
                with training set size {train_size} and validation set size {validate_size}"
        )
        train_dataset, validate_dataset = random_split(
            dataset=tv_dataset,
            lengths=[train_size, validate_size],
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        validate_dataloader = DataLoader(
            dataset=validate_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        dataloaders_dict = {"train": train_dataloader, "validate": validate_dataloader}
        return dataloaders_dict

    def train_ensemble_model(self, pretrained=True):
        '''
        训练整体集成模型，通过训练不同的单个模型来集成。
        各个模型通过random_split打乱训练集，训练出不同。
        '''
        val_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        train_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        ensem_range = range(self.ensem)

        for model_num in ensem_range:
            print(f"Start Training Ensem Number {model_num}")
            model_save_path = os.path.join(self.model_dir, f"checkpoint{model_num}.pth.tar")
            if os.path.exists(model_save_path) and pretrained:
                print("Found pretrained model {}".format(model_save_path))
                validate_metrics = torch.load(model_save_path, weights_only=True)
            else:
                dataloaders_dict = self.get_train_validate_dataloaders_dict()
                train_metrics, validate_metrics, model = self.train_single_model(
                    dataloaders_dict, model_save_path
                )
                for column in train_metrics.keys():
                    train_df.loc[model_num, column] = train_metrics[column]

            for column in validate_metrics.keys():
                if column == "model_state_dict":
                    continue
                val_df.loc[model_num, column] = validate_metrics[column]

        val_df = val_df.astype(float).round(3)
        val_df.loc["Mean"] = val_df.mean()
        val_path = os.path.join(self.exp_dir, f"validate_metrics-{self.model_obj.name}-ensem{self.ensem}.csv")
        val_df.to_csv(val_path, index=True)
        return

    def train_single_model(self, dataloaders_dict, model_save_path):
        '''
        训练单一模型
        '''
        model = self.model_obj.init_model(device=self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print("Training on device {} under {}".format(self.device, model_save_path))
        print(model)

        cudnn.benchmark = True
        start_time = time.time()
        best_validate_metrics = {"loss": 1000.0, "accy": 0.0, "MCC": 0.0, "epoch": 0, "diff": 0.0}
        best_model = copy.deepcopy(model.state_dict())
        train_metrics = {"prev_loss": 1000.0, "pattern_accy": -1}

        for epoch in range(self.max_epoch):
            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                
                data_iterator = tqdm(dataloaders_dict[phase], unit="batch")
                data_iterator.set_description("Epoch {}: {}".format(epoch, phase))

                running_metrics = {
                    "running_loss": 0.0,
                    "running_correct": 0.0,
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                }
                for i, batch in enumerate(data_iterator):
                    inputs = batch["image"].to(self.device, dtype=torch.float)
                    labels = batch["label"].to(self.device, dtype=self.label_dtype)
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = self.loss_from_model_output(labels, outputs)
                        _, preds = torch.max(outputs, 1)
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    self.update_running_metrics(loss, labels, preds, running_metrics)
                    del inputs, labels
                num_samples = len(dataloaders_dict[phase].dataset)
                epoch_stat = self.generate_epoch_stat(epoch, self.lr, num_samples, running_metrics)
                print(epoch_stat)

                if phase == "validate":
                    if epoch_stat["loss"] < best_validate_metrics["loss"]:
                        for metric in ["loss", "accy", "MCC", "epoch", "diff"]:
                            best_validate_metrics[metric] = epoch_stat[metric]
                        best_model = copy.deepcopy(model.state_dict())

            if self.early_stop and (epoch - best_validate_metrics["epoch"]) >= 2:
                break
            print()

        print("Training complete in {:.1f}s".format(time.time() - start_time))
        print("Best validate loss: {:4f} at epoch {}, ".format(
            best_validate_metrics["loss"], best_validate_metrics["epoch"]))

        model.load_state_dict(best_model)
        best_validate_metrics["model_state_dict"] = model.state_dict().copy()
        torch.save(best_validate_metrics, model_save_path)

        train_metrics = self.evaluate(model, {"train": dataloaders_dict["train"]})["train"]
        train_metrics["epoch"] = best_validate_metrics["epoch"]
        self.release_dataloader_memory(dataloaders_dict, model)
        del best_validate_metrics["model_state_dict"]
        return train_metrics, best_validate_metrics, model

    @staticmethod
    def update_running_metrics(loss, labels, preds, running_metrics):
        '''
        统计每个epoch的label预测指标
        '''
        running_metrics["running_loss"] += loss.item() * len(labels)
        running_metrics["running_correct"] += (preds == labels).sum().item()
        running_metrics["TP"] += (preds * labels).sum().item()
        running_metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
        running_metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
        running_metrics["FN"] += ((preds - 1) * labels).sum().abs().item()

    @staticmethod
    def generate_epoch_stat(epoch, learning_rate, num_samples, running_metrics):
        '''
        生成一次epoch的评价指标，epoch_stat: dict
        keys: epoch, lr, diff, loss, accy, MCC
        '''
        TP, TN, FP, FN = (
            running_metrics["TP"],
            running_metrics["TN"],
            running_metrics["FP"],
            running_metrics["FN"],
        )
        epoch_stat = {"epoch": epoch, "lr": "{:.2E}".format(learning_rate)}
        epoch_stat["diff"] = 1.0 * ((TP + FP) - (TN + FN)) / num_samples
        epoch_stat["loss"] = running_metrics["running_loss"] / num_samples
        epoch_stat["accy"] = 1.0 * running_metrics["running_correct"] / num_samples
        epoch_stat["MCC"] = (
            np.nan
            if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
            else 1.0
            * (TP * TN - FP * FN)
            / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        return epoch_stat

    def evaluate(self, model, dataloaders_dict):
        print("Evaluating model on device: {}".format(self.device))
        model.to(self.device)
        res_dict = {}
        for subset in dataloaders_dict.keys():
            start_time = time.time()
            data_iterator = tqdm(dataloaders_dict[subset], unit="batch")
            data_iterator.set_description("Evaluation: ")
            model.eval()
            running_metrics = {
                "running_loss": 0.0,
                "running_correct": 0.0,
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
            }
            for batch in data_iterator:
                inputs = batch["image"].to(self.device, dtype=torch.float)
                labels = batch["label"].to(self.device, dtype=self.label_dtype)
                outputs = model(inputs)
                loss = self.loss_from_model_output(labels, outputs)
                _, preds = torch.max(outputs, 1)
                self.update_running_metrics(loss, labels, preds, running_metrics)
                del inputs, labels
            num_samples = len(dataloaders_dict[subset].dataset)
            epoch_stat = self.generate_epoch_stat(-1, -1, num_samples, running_metrics)
            data_iterator.set_postfix(epoch_stat)
            data_iterator.update()
            print(epoch_stat)
            print("Evaluation on {} complete in {:.1f}s".format(subset, time.time() - start_time))
            res_dict[subset] = {metric: epoch_stat[metric] for metric in ["loss", "accy", "MCC", "diff"]}
        del model
        torch.cuda.empty_cache()
        return res_dict

    def loss_from_model_output(self, labels, outputs):
        if self.loss_name == "cross_entropy":
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        elif self.loss_name == "MSE":
            loss = torch.nn.MSELoss()(outputs.flatten(), labels)
        else:
            raise Exception("Unknown loss_name type")
        return loss

    @staticmethod
    def release_dataloader_memory(dataloaders_dict, model):
        for key in list(dataloaders_dict.keys()):
            dataloaders_dict[key] = None
        del model
        torch.cuda.empty_cache()


    def get_dataloader_for_year(self, year):
        year_dataset = ImageDataset(
            window_size=self.ws,
            predict_window_size=self.pw,
            year=year,
            has_volume_bar=self.has_volume_bar,
            ma_lag_list=self.ma_lag_list,
            annual_stocks_num=self.annual_stocks_num,
            tstat_threshold=self.tstat_threshold,
            regression_label=self.regression_label
        )
        year_dataloader = DataLoader(year_dataset, batch_size=BATCH_SIZE)
        return year_dataloader

    def generate_oos_ensemble_results(self, load_saved_data=True):
        '''
        生成ensem模型预测结果，保存为.csv
        '''
        year_list = list(self.oos_year_list)
        model_list = [self.model_obj.init_model() for _ in range(self.ensem)]
        for year in year_list:
            print()
            models_path_list = [os.path.join(self.model_dir, f"checkpoint{i}.pth.tar") for i in range(self.ensem)]
            ensem_res_path = os.path.join(self.ensem_res_dir, f"ensem{self.ensem}_res_{year}.csv")
            if os.path.exists(ensem_res_path) and load_saved_data:
                print(f"Found {ensem_res_path}")
                continue
            else:
                print(f"Generating {self.ws}d{self.pw}p ensem results for year {year} with freq {self.pf_freq}")
                print("Loading saved model from: {} to {}".format(models_path_list[0], models_path_list[-1]))

                for i, model in enumerate(model_list):
                    model_state_dict = torch.load(models_path_list[i],
                                                  map_location=self.device, weights_only=True)["model_state_dict"]
                    model.load_state_dict(model_state_dict)

                year_dataloader = self.get_dataloader_for_year(year)

                df = self.ensemble_results(model_list, year_dataloader)
                df.to_csv(ensem_res_path)

    def ensemble_results(self, model_list, dataloader):
        '''
        具体生成ensem的预测的df
        df.columns为["StockID", "ending_date", "up_prob", "ret_val", "MarketCap"]
        '''
        df_columns = ["StockID", "ending_date", "up_prob", "ret_val", "MarketCap"]
        df_dtypes = [object, "datetime64[ns]", np.float64, np.float64, np.float64]
        df_list = []
        for batch in dataloader:
            image = batch["image"].to(self.device, dtype=torch.float)
            if self.model_obj.regression_label is None:
                total_prob = torch.zeros(len(image), 2, device=self.device)
            else:
                total_prob = torch.zeros(len(image), 1, device=self.device)
            for model in model_list:
                model.to(self.device)
                model.eval()
                with torch.set_grad_enabled(False):
                    outputs = model(image)
                    if self.model_obj.regression_label is None:
                        outputs = nn.Softmax(dim=1)(outputs)
                total_prob += outputs
            del image
            batch_df = cfg.df_empty(df_columns, df_dtypes)
            batch_df["StockID"] = batch["StockID"]
            batch_df["ending_date"] = pd.to_datetime([str(t) for t in batch["ending_date"]])
            batch_df["ret_val"] = np.nan_to_num(batch["ret_val"].numpy()).reshape(-1)
            batch_df["MarketCap"] = np.nan_to_num(batch["MarketCap"].numpy()).reshape(-1)
            if self.model_obj.regression_label is None:
                batch_df["up_prob"] = total_prob[:, 1].cpu()
            else:
                batch_df["up_prob"] = total_prob.flatten().cpu()
            df_list.append(batch_df)
        df = pd.concat(df_list)
        df["up_prob"] = 1.0 * df["up_prob"] / len(model_list)
        df.reset_index(drop=True)
        return df

    def load_ensemble_results(self, year=None, multiindex=False):
        assert (year is None) or isinstance(year, int) or isinstance(year, list)
        year_list = (
            self.oos_year_list
            if year is None
            else [year]
            if isinstance(year, int)
            else year
        )
        df_list = []
        for y in year_list:
            ensem_res_path = os.path.join(self.ensem_res_dir, f"ensem{self.ensem}_res_{y}.csv")
            if os.path.exists(ensem_res_path):
                df = pd.read_csv(
                    ensem_res_path,
                    parse_dates=["ending_date"],
                    index_col=0
                )
                df.StockID = df.StockID.astype(str)
            else:
                self.generate_oos_ensemble_results()
                df = pd.read_csv(
                    ensem_res_path,
                    parse_dates=["ending_date"],
                    index_col=0,
                )
                df.StockID = df.StockID.astype(str)
            df_list.append(df)
        whole_ensemble_res = pd.concat(df_list, ignore_index=True)
        whole_ensemble_res.rename(columns={"ending_date": "Date"}, inplace=True)
        whole_ensemble_res.set_index(["Date", "StockID"], inplace=True)
        
        if not multiindex:
            whole_ensemble_res.reset_index(inplace=True, drop=False)
        whole_ensemble_res.dropna(inplace=True)
        return whole_ensemble_res

    def generate_oos_stat(self):
        def cross_entropy_loss(pred_prob, true_label):
            pred_prob = np.array(pred_prob)
            x = np.zeros((len(pred_prob), 2))
            x[:, 1] = pred_prob
            x[:, 0] = 1 - x[:, 1]
            pred_prob = x

            true_label = np.array(true_label)
            y = np.zeros((len(true_label), 2))
            y[np.arange(true_label.size), true_label] = 1
            true_label = y

            loss = -np.sum(true_label * np.log(pred_prob)) / len(pred_prob)
            return loss

        def calculate_test_log(pred_prob, label):
            pred = np.where(pred_prob > 0.5, 1, 0)
            num_samples = len(pred)
            TP = np.nansum(pred * label, dtype=np.int64) / num_samples
            TN = np.nansum((pred - 1) * (label - 1)) / num_samples
            FP = np.abs(np.nansum(pred * (label - 1))) / num_samples
            FN = np.abs(np.nansum((pred - 1) * label)) / num_samples
            test_log = {
                "diff": 1.0 * ((TP + FP) - (TN + FN)),
                "loss": cross_entropy_loss(pred_prob, label),
                "accy": 1.0 * (TP + TN),
                "MCC": np.nan
                if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
                else 1.0
                * (TP * TN - FP * FN)
                / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
            }
            return test_log

        ensem_res = self.load_ensemble_results(
            year=self.oos_year_list, multiindex=True
        )
        ret_name = f"ret_val"

        def _prob_and_ret_rank_corr(df):
            prob_rank = df["up_prob"].rank(method="average", ascending=False)
            ret_rank = df[ret_name].rank(method="average", ascending=False)
            return ret_rank.corr(prob_rank, method="spearman")

        def _prob_and_ret_pearson_corr(df):
            return pd.Series(df["up_prob"]).corr(df[ret_name], method="pearson")

        pred_prob = ensem_res.up_prob.to_numpy()
        label = np.where(ensem_res[ret_name].to_numpy() > 0, 1, 0)
        if self.model_obj.regression_label is not None:
            pred_prob += 0.5
        oos_metrics = calculate_test_log(pred_prob, label)

        rank_corr = ensem_res.groupby("Date").apply(_prob_and_ret_rank_corr)
        pearson_corr = ensem_res.groupby("Date").apply(_prob_and_ret_pearson_corr)
        oos_metrics["Spearman"] = rank_corr.mean()
        oos_metrics["Pearson"] = pearson_corr.mean()
        with open(self.oos_metrics_path, "wb+") as f:
            pickle.dump(oos_metrics, f)
        return oos_metrics

    def load_oos_stat(self):
        oos_metrics_path = self.oos_metrics_path

        if os.path.exists(oos_metrics_path):
            print(f"Loading oos metrics from {oos_metrics_path}")
            with open(oos_metrics_path, "rb") as f:
                oos_metrics = pickle.load(f)
        else:
            oos_metrics = self.generate_oos_stat()
        return oos_metrics

    def calculate_portfolio(self, cut=10, transaction_cost=False): 
        self.generate_oos_ensemble_results()

        # pf_obj = self.load_portfolio_obj(transaction_cost=transaction_cost)
        whole_ensemble_res = self.load_ensemble_results(self.oos_year_list, multiindex=True)
        pf_obj = PortfolioManager(
            signal_df=whole_ensemble_res,
            freq=self.pf_freq,
            portfolio_dir=self.portfolio_dir,
            year_list=self.oos_year_list,
            transaction_cost=transaction_cost,
        )
        pf_obj.generate_portfolio(cut=cut)



def Set_model(window_size, predict_window_size):

    exp_obj = Experiment(
        window_size=window_size,
        predict_window_size=predict_window_size,
        ensemble_nums=5,
        learning_rate=1e-5,
        drop_prob=0.5,
        device_number=0,
        max_epoch=30,
        early_stop=True,

        ma_lag_list=[20],
        has_volume_bar=True,

        loss_name="cross_entropy",
        train_size_ratio=0.7,
        annual_stocks_num="all",

        weight_decay=0,
        tstat_threshold=0,
    )

    torch.set_num_threads(1)

    exp_obj.train_ensemble_model()
    exp_obj.generate_oos_ensemble_results()
    # tmp = exp_obj.load_ensemble_results(multiindex=True)
    # print(tmp)

    # oos_stat = exp_obj.load_oos_stat()
    # print(oos_stat)

    exp_obj.calculate_portfolio(cut=10, transaction_cost=0.001)

    del exp_obj
    return

