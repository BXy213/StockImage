import os
import os.path as op
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import get_stock_data as gsd


class PortfolioManager(object):
    def __init__(self,
        signal_df: pd.DataFrame,
        freq,
        portfolio_dir,
        year_list,
        transaction_cost=False,
    ):
        assert freq in ["week", "month"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.year_list = year_list
        self.no_delay_ret_name = f"ret_val"
        self.transaction_cost = transaction_cost

        assert "up_prob" in signal_df.columns
        self.signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(self.year_list)]
        # self.signal_df.to_csv("D:\\Users\\vbxy2\\Desktop\\self_signal_df.csv")
        # print()

    def calculate_portfolio_rets(self, weight_type, cut=10):
        # 返回每日0组到9组的portfolio_ret.csv
        assert weight_type in ["ew", "vw"]
        ret_name = self.no_delay_ret_name
        df = self.signal_df.copy()

        # print("!!!")
        # print(df.head())
        # print("!!!")

        def __get_decile_df_with_inv_ret(reb_df, decile_idx):
            rebalance_up_prob = reb_df.up_prob
            low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = reb_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                _decile_df["weight"] = (
                    pd.Series(
                        np.ones(stock_num), dtype=np.float64, index=_decile_df.index
                    )
                    / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=np.float64) / np.sum(value)
            _decile_df["inv_ret"] = _decile_df["weight"] * _decile_df[ret_name]
            return _decile_df

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        print(f"Calculating portfolio from {dates[0]}, {dates[1]} to {dates[-1]}")
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None
        prob_ret_corr = []
        prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []

        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()
            rank_corr = cfg.rank_corr(
                rebalance_df, "up_prob", ret_name, method="spearman"
            )
            pearson_corr = cfg.rank_corr(
                rebalance_df, "up_prob", ret_name, method="pearson"
            )
            prob_ret_corr.append(rank_corr)
            prob_ret_pearson_corr.append(pearson_corr)

            low = np.percentile(rebalance_df.up_prob, 10)
            high = np.percentile(rebalance_df.up_prob, 90)
            if low == high:
                print(f"Skipping {date}")
                continue
            for j in range(cut):
                decile_df = __get_decile_df_with_inv_ret(rebalance_df, j)
                if self.transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                            decile_df["weight"] * self.transaction_cost
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                            decile_df["weight"] * self.transaction_cost
                        )
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = __get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = __get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            buy_sell_decile = pd.concat([sell_decile, buy_decile])
            prob_inv_ret_corr.append(
                cfg.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
            )
            prob_inv_ret_pearson_corr.append(
                cfg.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            )

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (
                -1
            )
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(
                    index=np.unique(list(to_df.index) + list(prev_to_df.index))
                )
                try:
                    tto_df["cur_weight"] = to_df["weight"]
                except ValueError:
                    pdb.set_trace()
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[
                    ["weight", ret_name, "inv_ret"]
                ]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (
                        tto_df["cur_weight"]
                        - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom
                    ).abs()
                )
                turnover[i - 1] = turnover[i - 1] * 0.5
            prev_to_df = to_df

        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        print(
            f"Spearman Corr between Prob and Stock Return is {np.nanmean(prob_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Stock Return is {np.nanmean(prob_ret_pearson_corr):.4f}"
        )
        print(
            f"Spearman Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_pearson_corr):.4f}"
        )
        
        return portfolio_ret, np.mean(turnover)

    @staticmethod
    def _ret_to_cum_log_ret(rets):
        log_rets = np.log(rets.astype(float) + 1)
        return log_rets.cumsum()

    def make_portfolio_plot(self, portfolio_ret, cut, weight_type, save_path, plot_title):
        print(
            f"Generating portfolio plot of {weight_type} at {save_path}"
        )
        ret_name = "next_freq_ewret" if weight_type == "ew" else "next_freq_vwret"
        df = portfolio_ret.copy()
        df.columns = ["Low(L)"] + [str(i) for i in range(2, cut)] + ["High(H)", "H-L"]
        mkt = gsd.get_mkt_freq_rets(self.freq)
        df["market"] = mkt[ret_name]
        df.dropna(inplace=True)
        top_col_name, bottom_col_name = ("High(H)", "Low(L)")
        log_ret_df = pd.DataFrame(index=df.index)
        for column in df.columns:
            log_ret_df[column] = self._ret_to_cum_log_ret(df[column])
        plt.figure()
        log_ret_df = log_ret_df[[top_col_name, bottom_col_name, "H-L", "market"]]
        prev_year = pd.to_datetime(log_ret_df.index[0]).year - 1
        prev_day = pd.to_datetime("{}-12-31".format(prev_year))
        log_ret_df.loc[prev_day] = [0, 0, 0, 0]
        plot = log_ret_df.plot(
            style={"market": "y", top_col_name: "b", bottom_col_name: "r", "H-L": "k"},
            lw=1,
            title=plot_title,
        )
        plot.legend(loc=2)
        plt.grid()
        plt.savefig(os.path.join(save_path, "pic+" + self.freq + "+" + weight_type))
        # plt.show()
        plt.close()

    def portfolio_res_summary(self, portfolio_ret, turnover, cut=10):
        # 这里获取年化收益率和标准差
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))
        period = 52 if self.freq == "week" else 12
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 / 4
            if self.freq == "week"
            else 1
            if self.freq == "month"
            else 3
        )
        print(summary_df)
        return summary_df

    def generate_portfolio(self, cut=10):
        # 注意在这里计算个股，所以需要在上面计算交易成本：self.calculate_portfolio_rets
        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, cut)
            print(f"Calculating {pf_name}")

            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type, cut=cut
            )
            data_dir = cfg.get_dir(op.join(self.portfolio_dir, "pf_data"))
            portfolio_ret.to_csv(op.join(data_dir, f"pf_data_{pf_name}.csv"))

            summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            print(f"Summary saved to {smry_path}")
            summary_df.to_csv(smry_path)

            # 作图，注意与market比较
            self.make_portfolio_plot(portfolio_ret, cut = cut, weight_type = weight_type, save_path = "D:\\Users\\vbxy2\\Desktop", plot_title = weight_type + "+" + self.freq)

    def get_portfolio_name(self, weight_type, cut):
        assert weight_type.lower() in ["ew", "vw"]
        cut_surfix = f"_{cut}cut"
        tc_surfix = "_notcost" if not self.transaction_cost else "_withtcost"
        pf_name = f"{weight_type.lower()}{cut_surfix}{tc_surfix}"
        return pf_name

