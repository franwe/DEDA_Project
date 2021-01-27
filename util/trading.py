import os
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
from datetime import datetime, timedelta
import pickle
from itertools import groupby
from operator import itemgetter
import pandas as pd

from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator

# from util.smoothing import bspline
from util.connect_db import connect_db, get_as_df
from util.density import hd_rnd_domain

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep


def _M_bounds_from_list(lst, df):
    groups = [
        [i for i, _ in group]
        for key, group in groupby(enumerate(lst), key=itemgetter(1))
        if key
    ]
    M_bounds = []
    for group in groups:
        M_bounds.append((df.M[group[0]], df.M[group[-1]]))
    return M_bounds


def get_buy_sell_bounds(rnd, hd, M, K_bound=0):
    K = rnd / hd
    df = pd.DataFrame({"M": M, "K": K})
    df["buy"] = df.K < (1 - K_bound)
    df["sell"] = df.K > (1 + K_bound)

    M_bounds_sell = _M_bounds_from_list(df.sell.tolist(), df)
    M_bounds_buy = _M_bounds_from_list(df.buy.tolist(), df)
    return M_bounds_sell, M_bounds_buy


def _get_payoff(K, ST, option):
    if option == "C":
        return max(ST - K, 0)
    elif option == "P":
        return max(K - ST, 0)


def execute_options(data, day, tau_day):
    eval_day = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=tau_day)
    db = connect_db()

    coll = db["BTCUSD_deribit"]
    query = {"date_str": str(eval_day.date())}
    ST = get_as_df(coll, query)["price"].iloc[0]

    query = {"date_str": day}
    S0 = get_as_df(coll, query)["price"].iloc[0]

    data["ST"] = ST
    data["opt_payoff"] = data.apply(
        lambda row: _get_payoff(row.K, ST, row.option), axis=1
    )
    print("--- S0: {} --- ST: {} --- M: {}".format(S0, ST, S0 / ST))
    return data


def _calculate_fee(P, S, max_fee_BTC=0.0004, max_fee_pct=0.2):
    option_bound = max_fee_pct * P
    underlying_bound = max_fee_BTC * S
    fee = min(underlying_bound, option_bound)
    return fee


def _trading_payoffs(data):
    buy_mask = data.action == "buy"

    data["trading_fee"] = data.apply(
        lambda row: _calculate_fee(row.P, row.S, max_fee_BTC=0.0004), axis=1
    )
    data["t0_payoff"] = data["P"]
    data.loc[buy_mask, "t0_payoff"] = -1 * data.loc[buy_mask, "P"]
    data["t0_payoff"] = data["t0_payoff"] - data["trading_fee"]

    data["T_payoff"] = -1 * data["opt_payoff"]
    data.loc[buy_mask, "T_payoff"] = +1 * data.loc[buy_mask, "opt_payoff"]
    data["delivery_fee"] = data.apply(
        lambda row: _calculate_fee(row.T_payoff, row.S, max_fee_BTC=0.0002),
        axis=1,
    )
    data.loc[~buy_mask, "delivery_fee"] = 0  # only applies to TAKER ORDERS
    data["T_payoff"] = data["T_payoff"] - data["delivery_fee"]

    data["total"] = data.t0_payoff + data.T_payoff
    return data


def add_results_to_table(
    df_results, results, trading_day, trading_tau, deviates_from_one_ratio
):
    if len(results) == 0:
        entry = {
            "date": trading_day,
            "tau_day": trading_tau,
            "t0_payoff": 0,
            "T_payoff": 0,
            "total": 0,
            "trade": "-",
            "kernel": deviates_from_one_ratio,
        }
        df_results = df_results.append(entry, ignore_index=True)

    else:
        for key in results:
            df_trades = results[key]

            entry = {
                "date": trading_day,
                "tau_day": trading_tau,
                "t0_payoff": df_trades.t0_payoff.sum(),
                "T_payoff": df_trades.T_payoff.sum(),
                "total": df_trades.total.sum(),
                "trade": key,
                "kernel": deviates_from_one_ratio,
            }
            df_results = df_results.append(entry, ignore_index=True)
    return df_results


# ---------------------------------------------------------- TRADING STRATEGIES


def options_in_interval(
    option, moneyness, action, df, left, right, near_bound
):
    if (moneyness == "ATM") & (option == "C"):
        mon_left = 1 - near_bound
        mon_right = 1 + near_bound
        which_element = 0
    elif (moneyness == "ATM") & (option == "P"):
        mon_left = 1 - near_bound
        mon_right = 1 + near_bound
        which_element = -1

    elif (moneyness == "OTM") & (option == "C"):
        mon_left = 0
        mon_right = 1 - near_bound
        which_element = -1
    elif (moneyness == "OTM") & (option == "P"):
        mon_left = 1 + near_bound
        mon_right = 10
        which_element = 0

    candidates = df[
        (df.M > left)
        & (df.M < right)  # option interval
        & (df.M > mon_left)
        & (df.M < mon_right)
        & (df.option == option)
    ]
    candidate = candidates.iloc[which_element]
    candidate["action"] = action
    return candidate


def K1(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        ["M", "option", "P", "K", "S", "iv", "P_BTC", "color", "opt_payoff"]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "buy"
    otm_put, otm_put_action = pd.Series(), "buy"
    atm_call, atm_call_action = pd.Series(), "sell"
    atm_put, atm_put_action = pd.Series(), "sell"

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            atm_call = options_in_interval(
                "C", "ATM", atm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            atm_put = options_in_interval(
                "P", "ATM", atm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, atm_call.empty, atm_put.empty, otm_put.empty]):
        pass
    else:
        df_trades = pd.DataFrame([otm_call, atm_call, atm_put, otm_put])
        return _trading_payoffs(df_trades)


def K2(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        ["M", "option", "P", "K", "S", "iv", "P_BTC", "color", "opt_payoff"]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "sell"
    otm_put, otm_put_action = pd.Series(), "sell"
    atm_call, atm_call_action = pd.Series(), "buy"
    atm_put, atm_put_action = pd.Series(), "buy"

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            atm_call = options_in_interval(
                "C", "ATM", atm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            atm_put = options_in_interval(
                "P", "ATM", atm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, atm_call.empty, atm_put.empty, otm_put.empty]):
        pass
    else:
        df_trades = pd.DataFrame([otm_call, atm_call, atm_put, otm_put])
        return _trading_payoffs(df_trades)


def S1(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        ["M", "option", "P", "K", "S", "iv", "P_BTC", "color", "opt_payoff"]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "buy"
    otm_put, otm_put_action = pd.Series(), "sell"

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, otm_put.empty]):
        pass
    elif (len(M_bounds_buy) > 1) or (len(M_bounds_sell) > 1):
        print(" ---- too many intervals")
        pass
    else:
        df_trades = pd.DataFrame([otm_call, otm_put])
        return _trading_payoffs(df_trades)


def S2(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        ["M", "option", "P", "K", "S", "iv", "P_BTC", "color", "opt_payoff"]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "sell"
    otm_put, otm_put_action = pd.Series(), "buy"

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, otm_put.empty]):
        pass
    elif (len(M_bounds_buy) > 1) or (len(M_bounds_sell) > 1):
        print(" ---- too many intervals")
        pass
    else:
        df_trades = pd.DataFrame([otm_call, otm_put])
        return _trading_payoffs(df_trades)


# ------------------------------------------------------- DATASTUFF - LOAD SAVE
def load_rnd_hd_from_pickle(data_path, day, tau_day):
    with open(data_path + "T-{}_{}.pkl".format(tau_day, day), "rb") as handle:
        data = pickle.load(handle)

    RND = data["RND"]
    hd, rnd, M = data["hd"], data["rnd"], data["M"]
    kernel = data["kernel"]
    return RND, hd, rnd, kernel, M


def save_trades_to_pickle(
    data_path,
    trading_day,
    trading_tau,
    rnd,
    rnd_points,
    hd,
    kernel,
    M,
    K_bound,
    M_bounds_buy,
    M_bounds_sell,
    df_all,
    results,
):
    if len(results) == 0:
        content = {
            "day": trading_day,
            "tau_day": trading_tau,
            "trade": "-",
            "rnd": rnd,
            "rnd_points": rnd_points,  # M, q, color
            "hd": hd,
            "kernel": kernel,
            "M": M,
            "K_bound": K_bound,
            "M_bounds_buy": M_bounds_buy,
            "M_bounds_sell": M_bounds_sell,
            "df_all": df_all,
            "df_trades": None,
        }

        filename = "T-{}_{}_{}.pkl".format(trading_tau, trading_day, "-")
        with open(data_path + filename, "wb") as handle:
            pickle.dump(content, handle)
        return
    else:
        for trade in results:
            df_trades = results[trade]
            content = {
                "day": trading_day,
                "tau_day": trading_tau,
                "trade": trade,
                "rnd": rnd,
                "rnd_points": rnd_points,  # M, q, color
                "hd": hd,
                "kernel": kernel,
                "M": M,
                "K_bound": K_bound,
                "M_bounds_buy": M_bounds_buy,
                "M_bounds_sell": M_bounds_sell,
                "df_all": df_all,
                "df_trades": df_trades,
            }

            filename = "T-{}_{}_{}.pkl".format(trading_tau, trading_day, trade)
            with open(data_path + filename, "wb") as handle:
                pickle.dump(content, handle)
    return


def load_trades_from_pickle(data_path, day, tau_day, trade):
    with open(
        data_path + "T-{}_{}_{}.pkl".format(tau_day, day, trade), "rb"
    ) as handle:
        data = pickle.load(handle)
    return data


# --------------------------------------------------------------- PLOT STRATEGY
def plot_strategy(
    M, kernel, df_tau, df_trades, K_bound, M_bounds_sell, M_bounds_buy, x=0.5
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(
        df_tau.M,
        [1] * len(df_tau.M),
        c=df_tau.color,
        marker="|",
        s=10,
        alpha=0.5,
    )
    ax.scatter(df_trades.M, [1] * len(df_trades.M), c=df_trades.color, s=20)
    ax.plot(M, kernel)
    ax.axhspan(1 - K_bound, 1 + K_bound, color="grey", alpha=0.1)
    for interval in M_bounds_buy:
        ax.axvspan(interval[0], interval[1], color="blue", alpha=0.1)
    for interval in M_bounds_sell:
        ax.axvspan(interval[0], interval[1], color="red", alpha=0.1)
    ax.set_xlim((1 - x), (1 + x))
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Kernel = rnd/hd")
    ax.set_ylim(0, 2)
    return fig