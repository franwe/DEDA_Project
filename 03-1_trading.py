import os
from os.path import join
import pandas as pd
import logging
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename="trading.log", level=logging.ERROR)

from util.data import RndDataClass
from util.general import create_dates, load_tau_section_parameters
from util.trading import (
    load_rnd_hd_from_pickle,
    create_intervals,
    add_action,
    execute_options,
    general_trading_payoffs,
    skewness_trade,
    kurtosis_trade,
    trade_results,
    create_option_trade_table,
    save_trades_to_pickle,
)

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
density_data = join(cwd, "data", "02-3_rnd_hd") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
n = 1  # None
(
    filename,
    near_bound,
    far_bound,
    h,
    tau_min,
    tau_max,
) = load_tau_section_parameters("big")
df = pd.DataFrame()
RndData = RndDataClass(cutoff=x)

days = create_dates(start="2020-03-01", end="2020-09-30")
for day in days:
    taus = RndData.analyse(day)
    for tau in taus:
        found_trades = False
        tau_day = tau["_id"]
        if (tau_day > tau_min) & (tau_day <= tau_max):

            try:
                print(tau_day, day)
                RND, hd, rnd, kernel, M, trade_table = load_rnd_hd_from_pickle(
                    density_data, day, tau_day
                )
            except FileNotFoundError:
                logging.info(day, tau_day, " ---- densities do not exist")
                break

            # ----------------------------------------- COMPARE HD RND - ACTION
            buy = rnd < hd
            intervals = create_intervals(buy, M)
            RND = add_action(RND, intervals)

            # ----------------------------------------- KERNEL DEVIATES FROM 1?
            K_bound = 0.3
            around_one = (kernel > 1 - K_bound) & (kernel < 1 + K_bound)
            deviates_from_one_ratio = 1 - around_one.sum() / len(kernel)

            try:
                RND = execute_options(RND, day, tau_day)
                RND = general_trading_payoffs(RND)
            except KeyError:
                logging.info(day, tau_day, " ---- Maturity not reached yet")

                break

            trades_list = []
            trade_entry_blanc = {
                "date": day,
                "tau_day": tau_day,
                "kernel": deviates_from_one_ratio,
                "trade": "-",
                "total": 0,
                "t0_payoff": 0,
                "T_payoff": 0,
            }

            for S_trade in ["S1", "S2"]:
                try:
                    trade_calls, trade_puts = skewness_trade(
                        RND, far_bound, trade_type=S_trade
                    )
                    possible_trades = [trade_calls, trade_puts]
                    result, single_trades = trade_results(possible_trades, n)

                    trade_entry = deepcopy(trade_entry_blanc)
                    trade_entry.update(
                        {
                            "trade": S_trade,
                            "total": result.total,
                            "t0_payoff": result.t0_payoff,
                            "T_payoff": result.T_payoff,
                        }
                    )
                    trades_list.append(trade_entry)
                    trade_table.update(
                        {
                            S_trade: create_option_trade_table(
                                RND, single_trades
                            )
                        }
                    )

                except ValueError:
                    logging.info(
                        day,
                        tau_day,
                        S_trade,
                        " ---- Trade can't be created",
                    )

                except AttributeError:
                    logging.info(
                        day,
                        tau_day,
                        S_trade,
                        " ---- Trade can't be created",
                    )

            for K_trade in ["K1", "K2"]:
                try:
                    (
                        otm_call_trades,
                        atm_call_trades,
                        atm_put_trades,
                        otm_put_trades,
                    ) = kurtosis_trade(
                        RND, near_bound, far_bound, trade_type=K_trade
                    )
                    atm_trades = [atm_call_trades, atm_put_trades]
                    otm_trades = [otm_call_trades, otm_put_trades]

                    atm_result, single_atm_trades = trade_results(
                        atm_trades, n
                    )
                    otm_result, single_otm_trades = trade_results(
                        otm_trades, n
                    )
                    single_trades = single_atm_trades.append(
                        single_otm_trades, ignore_index=True
                    )

                    if (otm_result is None) or (atm_result is None):
                        pass
                    else:
                        result = otm_result + atm_result
                        trade_entry = deepcopy(trade_entry_blanc)
                        trade_entry.update(
                            {
                                "trade": K_trade,
                                "total": result.total,
                                "t0_payoff": result.t0_payoff,
                                "T_payoff": result.T_payoff,
                            }
                        )
                        trades_list.append(trade_entry)
                        trade_table.update(
                            {
                                K_trade: create_option_trade_table(
                                    RND, single_trades
                                )
                            }
                        )

                except ValueError:
                    logging.info(
                        day,
                        tau_day,
                        K_trade,
                        " ---- Trade can't be created",
                    )

                except AttributeError:
                    logging.info(
                        day,
                        tau_day,
                        K_trade,
                        " ---- Trade can't be created",
                    )

            if len(trades_list) > 0:
                df = df.append(trades_list, ignore_index=True)
                print(
                    day,
                    tau_day,
                    "-------",
                    pd.DataFrame(trades_list).trade.tolist(),
                )

            else:
                df = df.append([trade_entry_blanc], ignore_index=True)
                trade_table.update(
                    {"-": create_option_trade_table(RND, pd.DataFrame())}
                )
                print(day, tau_day, "------- ----------")

            save_trades_to_pickle(
                save_data, day, tau_day, rnd, hd, kernel, M, trade_table
            )


df[
    ["date", "tau_day", "total", "t0_payoff", "T_payoff", "trade", "kernel"]
].to_csv(save_data + filename, index=False)
