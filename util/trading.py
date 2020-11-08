import os
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
from datetime import datetime, timedelta

from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator
from util.smoothing import bspline
from util.connect_db import connect_db, get_as_df
from util.density import hd_rnd_domain

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep


def get_densities(
    RndData,
    HdData,
    day,
    tau_day,
    x,
    y_lim=None,
    reset_S=False,
    overwrite=False,
    h_densfit=0.15,
    cutoff=0.5,
):

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="complete")

    hd_data, S0 = HdData.filter_data(day)
    print(S0, day, tau_day)
    if reset_S:
        df_tau["S"] = S0

    df_tau["M"] = df_tau.S / df_tau.K

    RND = RndCalculator(df_tau, tau_day, day, h_densfit=h_densfit)
    RND.fit_smile()
    RND.rookley()

    HD = HdCalculator(
        data=hd_data,
        S0=S0,
        path=garch_data,
        tau_day=tau_day,
        date=day,
        cutoff=cutoff,
        n=400,
        M=5000,
        overwrite=overwrite,
    )
    HD.get_hd(variate=True)
    return HD, RND


def plot_hd_rnd(HD, RND, x, day, tau_day):
    hd, rnd, M = hd_rnd_domain(
        # HD,
        # RND,
        # interval=[1 - x, 1 + x]
        HD,
        RND,
        interval=[RND.data.M.min() * 0.9, RND.data.M.max() * 1.1],
    )
    calls = RND.data[RND.data.option == "C"]
    puts = RND.data[RND.data.option == "P"]

    # -------------------------------------------------------------------- Plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot(HD.M, HD.q_M, "b")
    ax.plot(RND.M, RND.q_M, "r")
    ax.plot(M, hd, "b", ls=":")
    ax.plot(M, rnd, "r", ls=":")
    ax.scatter(calls.M, calls.q_M, 5, c="r", label="calls")
    ax.scatter(puts.M, puts.q_M, 5, c="b", label="puts")

    ax.text(
        0.99,
        0.99,
        str(day) + "\n" + r"$\tau$ = " + str(tau_day),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_ylim(0)
    ax.set_xlim((1 - x), (1 + x))
    return fig, hd, rnd, M


def create_intervals(sequence, M):
    last = sequence[0]
    intervals = []
    this_interval = [last, M[0]]
    for i, tf in enumerate(sequence):
        if tf == last:
            pass
        else:
            this_interval.append(M[i - 1])
            intervals.append(this_interval)
            # set up new interval
            last = tf
            this_interval = [last, M[i]]
    this_interval.append(M[i])
    intervals.append(this_interval)
    return intervals


def add_action(RND, intervals):
    """add a column with trader's action to the RND dataframe

    Args:
        RND (class): RND class with object RND.data dataframe
        intervals (list): list of intervals with [bool, start_idx, end_idx]
                          True: 'buy', False: 'sell'
        M (array): array of moneyness values, belonging to start_idx/end_idx

    Returns:
        class: RND class with new column RND.data.action 'buy'/'sell'
    """
    RND.data["action"] = None
    for interval in intervals:
        tf, start, end = interval
        mask = (RND.data.M >= start) & (RND.data.M < end)
        if tf == True:
            RND.data.loc[mask, "action"] = "buy"
        elif tf == False:
            RND.data.loc[mask, "action"] = "sell"
    return RND


def get_payoff(K, ST, option):
    if option == "C":
        return max(ST - K, 0)
    elif option == "P":
        return max(K - ST, 0)


def execute_options(RND, day, tau_day):
    eval_day = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=tau_day)
    db = connect_db()

    coll = db["BTCUSD_deribit"]
    query = {"date_str": str(eval_day.date())}
    ST = get_as_df(coll, query)["price"].iloc[0]

    query = {"date_str": day}
    S0 = get_as_df(coll, query)["price"].iloc[0]

    RND.data["opt_payoff"] = RND.data.apply(
        lambda row: get_payoff(row.K, ST, row.option), axis=1
    )
    print(S0, ST, S0 / ST)
    return RND


def general_trading_payoffs(RND):
    buy_mask = RND.data.action == "buy"

    RND.data["t0_payoff"] = RND.data["P"]
    RND.data.loc[buy_mask, "t0_payoff"] = -1 * RND.data.loc[buy_mask, "P"]

    RND.data["T_payoff"] = -1 * RND.data["opt_payoff"]
    RND.data.loc[buy_mask, "T_payoff"] = (
        +1 * RND.data.loc[buy_mask, "opt_payoff"]
    )

    RND.data["total"] = RND.data.t0_payoff + RND.data.T_payoff
    return RND


def create_trading_mask(RND, option, action, M_interval):
    trading_mask = (
        (RND.data.option == option)
        & (RND.data.action == action)
        & (RND.data.M > M_interval[0])
        & (RND.data.M <= M_interval[1])
    )
    return trading_mask


def get_trades_from_df(RND, trading_mask):
    trades = (
        RND.data[trading_mask][["t0_payoff", "T_payoff", "total", "K"]]
        .groupby(by="K")
        .mean()
    )
    return trades


def skewness_trade(RND, far_bound, trade_type="S1"):
    if trade_type == "S1":
        call_action = "buy"
        put_action = "sell"
    elif trade_type == "S2":
        call_action = "sell"
        put_action = "buy"

    far_call_mask = create_trading_mask(
        RND, "C", call_action, [0, 1 - far_bound]
    )
    far_put_mask = create_trading_mask(
        RND, "P", put_action, [1 + far_bound, 2]
    )

    trade_calls = get_trades_from_df(RND, far_call_mask)
    trade_puts = get_trades_from_df(RND, far_put_mask)
    return trade_calls, trade_puts


def trade_results(possible_trades, n=None):
    shapes = []
    for trades in possible_trades:
        shape = trades.shape[0]
        shapes.append(shape)
    n_max = min(shapes)  # max possible n to have symmetric trade

    if n_max == 0:
        return None, None

    if n is None:
        n = n_max
    else:
        n = min(n, n_max)

    trades_0 = possible_trades[0].iloc[0:n]
    trades_1 = possible_trades[1].iloc[-n:]

    result = trades_0.sum() + trades_1.sum()
    K = trades_0.index.tolist() + trades_1.index.tolist()
    return result, K


def kurtosis_trade(RND, near_bound, far_bound, trade_type="K1"):
    if trade_type == "K1":
        atm_action = "sell"
        otm_action = "buy"
    elif trade_type == "K2":
        atm_action = "buy"
        otm_action = "sell"

    otm_call_mask = create_trading_mask(
        # RND, "C", otm_action, [1 - far_bound, 1 - near_bound]
        RND,
        "C",
        otm_action,
        [0, 1 - near_bound],
    )
    atm_call_mask = create_trading_mask(
        RND, "C", atm_action, [1 - near_bound, 1]
    )
    atm_put_mask = create_trading_mask(
        RND, "P", atm_action, [1, 1 + near_bound]
    )
    otm_put_mask = create_trading_mask(
        # RND, "P", otm_action, [1 + near_bound, 1 + far_bound]
        RND,
        "P",
        otm_action,
        [1 + near_bound, 2],
    )

    otm_call_trades = get_trades_from_df(RND, otm_call_mask)
    atm_call_trades = get_trades_from_df(RND, atm_call_mask)
    atm_put_trades = get_trades_from_df(RND, atm_put_mask)
    otm_put_trades = get_trades_from_df(RND, otm_put_mask)

    return otm_call_trades, atm_call_trades, atm_put_trades, otm_put_trades
