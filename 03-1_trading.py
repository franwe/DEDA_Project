import os
from os.path import join
import pandas as pd
import logging
from copy import deepcopy

logging.basicConfig(filename="trading.log")  # , level=logging.DEBUG)

from util.data import RndDataClass, HdDataClass
from util.trading import (
    get_densities,
    create_intervals,
    add_action,
    execute_options,
    general_trading_payoffs,
    skewness_trade,
    kurtosis_trade,
    trade_results,
)
from util.density import hd_rnd_domain

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
n = 1  # None


def load_tau_section_parameters(tau_section):
    if tau_section == "small":
        return "trades_smallTau.csv", 0.1, 0.3, 0.15, 7, 40
    elif tau_section == "big":
        return "trades_bigTau.csv", 0.15, 0.35, 0.25, 40, 99
    elif tau_section == "huge":
        return "trades_hugeTau.csv", 0.2, 0.4, 0.35, 99, 182


(
    filename,
    near_bound,
    far_bound,
    h,
    tau_min,
    tau_max,
) = load_tau_section_parameters("huge")

HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
df = pd.DataFrame()


def create_dates(start, end):
    dates = pd.date_range(start, end, closed="right", freq="D")
    return [str(date.date()) for date in dates]


days = create_dates(start="2020-03-01", end="2020-09-30")
for day in days:
    taus = RndData.analyse(day)
    for tau in taus:
        found_trades = False
        tau_day = tau["_id"]
        if (tau_day > tau_min) & (tau_day <= tau_max):

            try:
                HD, RND = get_densities(
                    RndData,
                    HdData,
                    day,
                    tau_day,
                    x=x,
                    reset_S=False,  # to build +/- Moneyness pairs (0.8, 1.2)
                    overwrite=False,
                    h_densfit=h,
                    cutoff=x,
                )
            except ValueError:
                logging.info(day, tau_day, " ---- something wrong with RND")
                # logging.info(e)
                break
            except ZeroDivisionError:
                logging.info(
                    day, tau_day, " ---- No data for this day, maturity"
                )
                # logging.info(e)
                break

            # ------------------------------------------------------------ PLOT
            hd, rnd, M = hd_rnd_domain(
                HD,
                RND,
                interval=[RND.data.M.min() * 0.9, RND.data.M.max() * 1.1],
            )

            # ----------------------------------------- COMPARE HD RND - ACTION
            buy = rnd < hd
            intervals = create_intervals(buy, M)
            RND = add_action(RND, intervals)

            # ----------------------------------------- KERNEL DEVIATES FROM 1?
            Kernel = rnd / hd
            K_bound = 0.3
            around_one = (Kernel > 1 - K_bound) & (Kernel < 1 + K_bound)
            deviates_from_one_ratio = 1 - around_one.sum() / len(Kernel)

            try:
                RND = execute_options(RND, day, tau_day)
                RND = general_trading_payoffs(RND)
            except KeyError:
                logging.info(day, tau_day, " ---- Maturity not reached yet")
                # logging.info(e)
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
                "K": "-",
            }

            for S_trade in ["S1", "S2"]:
                try:
                    trade_calls, trade_puts = skewness_trade(
                        RND, far_bound, trade_type=S_trade
                    )
                    possible_trades = [trade_calls, trade_puts]
                    result, K = trade_results(possible_trades, n)

                    trade_entry = deepcopy(trade_entry_blanc)
                    trade_entry.update(
                        {
                            "trade": S_trade,
                            "total": result.total,
                            "t0_payoff": result.t0_payoff,
                            "T_payoff": result.T_payoff,
                            "K": K,
                        }
                    )
                    trades_list.append(trade_entry)

                except ValueError:
                    logging.info(
                        day,
                        tau_day,
                        S_trade,
                        " ---- Trade can't be created",
                    )
                    # logging.info(e)
                except AttributeError:
                    logging.info(
                        day,
                        tau_day,
                        S_trade,
                        " ---- Trade can't be created",
                    )
                    # logging.info(e)

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

                    atm_result, atm_K = trade_results(atm_trades, n)
                    otm_result, otm_K = trade_results(otm_trades, n)

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
                                "K": atm_K + otm_K,
                            }
                        )
                        trades_list.append(trade_entry)

                except ValueError:
                    logging.info(
                        day,
                        tau_day,
                        K_trade,
                        " ---- Trade can't be created",
                    )
                    # logging.info(e)
                except AttributeError:
                    logging.info(
                        day,
                        tau_day,
                        K_trade,
                        " ---- Trade can't be created",
                    )
                    # logging.info(e)

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
                print(day, tau_day, "------- ----------")


df[
    ["date", "tau_day", "total", "t0_payoff", "T_payoff", "trade", "kernel"]
].to_csv(save_data + filename, index=False)
