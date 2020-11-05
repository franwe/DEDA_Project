import os
from matplotlib import pyplot as plt
from os.path import join
import pandas as pd
import logging

logging.basicConfig(filename="trading.log", level=logging.DEBUG)

from util.data import RndDataClass, HdDataClass
from util.trading import (
    get_densities,
    plot_hd_rnd,
    create_intervals,
    add_action,
    execute_options,
    general_trading_payoffs,
    skewness_trade,
    kurtosis_trade,
    trade_results,
)

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.7
n = None
n = 1
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
df = pd.DataFrame()

day = "2020-04-07"  # 52  - S
day = "2020-07-14"  # 17  -
day = "2020-04-05"  # 54  -
day = "2020-04-26"  # 61  - S
day = "2020-07-14"  # 17  - K
day = "2020-07-12"  # 19  -
day = "2020-06-26"  # 63  - K
day = "2020-03-11"  # 9   -
day = "2020-07-05"  # 54  - K
day = "2020-05-04"  # 53  - S
tau_day = 53
print(RndData.analyse(day))


def create_dates(start, end):
    dates = pd.date_range(start, end, closed="right", freq="D")
    return [str(date.date()) for date in dates]


days = create_dates(start="2020-03-01", end="2020-09-30")
for day in days:
    print(day)
    taus = RndData.analyse(day)
    for tau in taus:
        tau_day = tau["_id"]
        # if (tau_day > 7) & (tau_day <= 40):  # h_densfit  = 0.15
        if (tau_day > 40) & (tau_day <= 99):  # h_densfit = 0.25
            near_bound = 0.1
            far_bound = 0.6

            try:
                HD, RND = get_densities(
                    RndData,
                    HdData,
                    day,
                    tau_day,
                    x=x,
                    reset_S=False,  # to build +/- Moneyness pairs (0.8, 1.2)
                    overwrite=False,
                    h_densfit=0.25,
                    cutoff=x,
                )
            except ValueError as e:
                logging.error("something wrong with RND: ", day, tau_day)
                logging.error(e)
                break

            ########################################################################## PLOT
            fig, hd, rnd, M = plot_hd_rnd(HD, RND, x, day, tau_day)

            ####################################################### COMPARE HD RND - ACTION
            buy = rnd < hd
            intervals = create_intervals(buy, M)
            RND = add_action(RND, intervals)
            try:
                RND = execute_options(RND, day, tau_day)
                RND = general_trading_payoffs(RND)
            except KeyError:
                logging.warning("Maturity not reached yet: ", day, tau_day)
                break

            for i in [1, 2]:
                S_trade = "S" + str(i)
                K_trade = "K" + str(i)

                try:
                    print(
                        "----------------------------------------------- {}".format(
                            S_trade
                        )
                    )
                    # TODO: different prices (S0) and therefore T_payoffs --> total. Take mean for now, but later
                    #       take min/max for "worst case result"
                    # TODO: transaction costs

                    # dynamic far_bound - only works for S1 "False" and S2 "True", does NOT work for K-trades
                    # far_bound = M[buy.tolist().index(False)] - 1

                    trade_calls, trade_puts = skewness_trade(
                        RND, near_bound, trade_type=S_trade
                    )
                    possible_trades = [trade_calls, trade_puts]
                    s_result, K = trade_results(possible_trades, n)
                    plt.vlines(
                        HD.S0 / K, ymin=0, ymax=2, linestyles="-", colors="r", alpha=0.3
                    )
                except ValueError as e:
                    print(e)
                    s_reult, K = [], []

                try:
                    print(
                        "----------------------------------------------- {}".format(
                            K_trade
                        )
                    )
                    (
                        otm_call_trades,
                        atm_call_trades,
                        atm_put_trades,
                        otm_put_trades,
                    ) = kurtosis_trade(RND, near_bound, far_bound, trade_type=K_trade)
                    atm_trades = [atm_call_trades, atm_put_trades]
                    atm_result, atm_K = trade_results(atm_trades, n)
                    plt.vlines(
                        HD.S0 / atm_K,
                        ymin=0,
                        ymax=2,
                        linestyles=":",
                        colors="c",
                        alpha=0.5,
                    )
                    otm_trades = [otm_call_trades, otm_put_trades]
                    otm_result, otm_K = trade_results(otm_trades, n)
                    plt.vlines(
                        HD.S0 / otm_K,
                        ymin=0,
                        ymax=2,
                        linestyles=":",
                        colors="b",
                        alpha=0.5,
                    )
                    k_result = otm_result + atm_result
                except ValueError as e:
                    print(e)
                    k_result, atm_K, otm_k = [], [], []

                add_info = pd.Series(
                    [day, tau_day, S_trade], index=["date", "tau_day", "trade"]
                )
                s_result = s_result.append(add_info)
                df = df.append(s_result, ignore_index=True)

                add_info = pd.Series(
                    [day, tau_day, K_trade], index=["date", "tau_day", "trade"]
                )
                k_result = k_result.append(add_info)
                df = df.append(k_result, ignore_index=True)

            print(day, tau_day)

df[["date", "tau_day", "trade", "t0_payoff", "T_payoff", "total"]].to_csv(
    save_data + "trades_bigTau.csv"
)

# # --------------------------------------------------------------------- ANALYZE
# k = (
#     RND.data[["M", "K", "t0_payoff", "total", "option", "action"]]
#     .groupby(by=["K", "option"])
#     .mean()
#     .reset_index()
# )

# mask = k.K.isin(atm_K)  # both options
# k["trade"] = 0
# k.loc[mask, "trade"] = "Axx"
# mask = k.K.isin(otm_K)  # both options
# k.loc[mask, "trade"] = "Oxx"

# print(k)

# s = (
#     RND.data[["M", "K", "t0_payoff", "total", "option", "action"]]
#     .groupby(by=["K", "option"])
#     .mean()
#     .reset_index()
# )
# mask = s.K.isin(K)  # both options
# s["trade"] = 0
# s.loc[mask, "trade"] = "XXX"

# print(s)

# plt.show()
