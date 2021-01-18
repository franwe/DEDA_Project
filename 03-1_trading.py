import os
from os.path import join
import pandas as pd
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename="trading.log", level=logging.INFO)

from util.data import RndDataClass
from util.general import create_dates, load_tau_section_parameters, add_one_day
from util.trading import (
    load_rnd_hd_from_pickle,
    get_buy_sell_bounds,
    execute_options,
    add_results_to_table,
    # plot_strategy,
    save_trades_to_pickle,
    K1,
    K2,
    S1,
    S2,
)

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
density_data = join(cwd, "data", "02-3_rnd_hd") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
(
    filename,
    near_bound,
    far_bound,
    h,
    tau_min,
    tau_max,
) = load_tau_section_parameters("small")
df_results = pd.DataFrame(
    columns=[
        "date",
        "tau_day",
        "t0_payoff",
        "T_payoff",
        "total",
        "trade",
        "kernel",
    ]
)
RndData = RndDataClass(cutoff=x)

tau_day = 12
day = "2020-03-22"  # today --> on 2020-03-23 can buy 13

days = create_dates(start="2020-03-01", end="2020-09-30")
for day in days:
    taus = RndData.analyse(day)
    for tau in taus:
        tau_day = tau["_id"]
        if (tau_day > tau_min) & (tau_day <= tau_max):
            try:
                print(tau_day, day)

                (
                    RND_strategy,
                    hd,
                    rnd,
                    kernel,
                    M,
                ) = load_rnd_hd_from_pickle(density_data, day, tau_day)

                call_mask = RND_strategy.data.option == "C"
                RND_strategy.data["color"] = "blue"  # blue - put
                RND_strategy.data.loc[call_mask, "color"] = "red"  # red - call
                rnd_points = RND_strategy.data[["M", "q_M", "color"]]

            except FileNotFoundError:
                logging.info(day, tau_day, " ---- densities do not exist")
                break

            # ---------------------------------- BUY-SELL INTERVALS FROM KERNEL
            K_bound = 0.3
            M_bounds_sell, M_bounds_buy = get_buy_sell_bounds(
                rnd, hd, M, K_bound
            )

            # -------------------------- LOAD OPTIONS THAT ARE OFFERED NEXT DAY
            trading_day = add_one_day(day)
            trading_tau = tau_day - 1

            RndData = RndDataClass(cutoff=x)
            df_tau = RndData.filter_data(
                date=trading_day, tau_day=trading_tau, mode="unique"
            )
            df_tau = execute_options(df_tau, trading_day, trading_tau)

            call_mask = df_tau.option == "C"
            df_tau["color"] = "blue"  # blue - put
            df_tau.loc[call_mask, "color"] = "red"  # red - call

            # ------------------------- TRY IF FIND RESULT FOR TRADING STRATEGY
            results = {}
            for name, strategy in zip(
                ["S1", "S2", "K1", "K2"], [S1, S2, K1, K2]
            ):
                try:
                    df_trades = strategy(df_tau, M_bounds_buy, M_bounds_sell)
                    results.update({name: df_trades})
                except IndexError:
                    pass

            print(trading_day, trading_tau, results.keys())
            # ----------------------------------------- KERNEL DEVIATES FROM 1?
            around_one = (kernel > 1 - K_bound) & (kernel < 1 + K_bound)
            deviates_from_one_ratio = 1 - around_one.sum() / len(kernel)

            df_results = add_results_to_table(
                df_results,
                results,
                trading_day,
                trading_tau,
                deviates_from_one_ratio,
            )
            save_trades_to_pickle(
                save_data,
                trading_day,
                trading_tau,
                rnd,
                rnd_points,  # M, q, color
                hd,
                kernel,
                M,
                K_bound,
                M_bounds_buy,
                M_bounds_sell,
                df_tau,
                results,
            )

            # try:
            #     RND = execute_options(RND, day, tau_day)
            #     RND = general_trading_payoffs(RND)
            # except KeyError:
            #     logging.info(day, tau_day, " ---- Maturity not reached yet")

            #     break


df_results.to_csv(save_data + filename, index=False)
