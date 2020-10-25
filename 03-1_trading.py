import os
from matplotlib import pyplot as plt
from os.path import join
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# --------------------------------------------------------------------- 2D PLOT


def get_densities(
    RndData,
    HdData,
    day,
    tau_day,
    x,
    y_lim=None,
    reset_S=False,
    overwrite=False,
    h_densfit=0.2,
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
        n=400,
        M=5000,
        overwrite=overwrite,
    )
    HD.get_hd(variate=True)
    return HD, RND


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff?


day = "2020-04-18"
tau_day = 41

HD, RND = get_densities(
    RndData,
    HdData,
    day,
    tau_day,
    x=x,
    reset_S=True,  # to build +/- Moneyness pairs (0.8, 1.2)
    overwrite=False,
    h_densfit=0.15,
)

from util.smoothing import bspline

_, HD_spline, _ = bspline(HD.M, HD.q_M, sections=15, degree=2)
_, RND_spline, _ = bspline(RND.M, RND.q_M, sections=15, degree=2)
M = np.linspace(min(min(RND.M), min(HD.M)), max(max(RND.M), max(HD.M)), 100)

hd = HD_spline(M)
rnd = RND_spline(M)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# --------------------------------------------------- Moneyness - Moneyness

ax.plot(HD.M, HD.q_M, "b")
ax.plot(RND.M, RND.q_M, "r")
ax.plot(M, hd, "b", ls=":")
ax.plot(M, rnd, "r", ls=":")
ax.text(
    0.99,
    0.99,
    str(day) + "\n" + r"$\tau$ = " + str(tau_day),
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.set_xlim((1 - x), (1 + x))
plt.show()


def create_intervals(sequence):
    last = sequence[0]
    intervals = []
    this_interval = [last, 0]
    for i, tf in enumerate(sequence):
        if tf == last:
            pass
        else:
            this_interval.append(i - 1)
            intervals.append(this_interval)
            # set up new interval
            last = tf
            this_interval = [last, i]
    this_interval.append(i)
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
        mask = (RND.data.M >= M[start]) & (RND.data.M < M[end])
        if tf == True:
            RND.data.loc[mask, "action"] = "buy"
        elif tf == False:
            RND.data.loc[mask, "action"] = "sell"
    return RND


buy = rnd < hd
intervals = create_intervals(buy)
RND = add_action(RND, intervals)

cols = ["M", "K", "S", "P", "P_BTC", "action", "option"]
RND.data[cols]
mask = (RND.data.option == "C") & (RND.data.action == "buy")
a = RND.data[mask][cols]
a.groupby(by="K").count()

eval_day = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=tau_day)
str(eval_day.date())

from util.connect_db import connect_db, get_as_df

db = connect_db()

coll = db["BTCUSD_deribit"]
query = {"date_str": str(eval_day.date())}
ST = get_as_df(coll, query)["price"].iloc[0]


def get_payoff(K, ST, option):
    if option == "C":
        return max(ST - K, 0)
    elif option == "P":
        return max(K - ST, 0)


RND.data["opt_payoff"] = RND.data.apply(
    lambda row: get_payoff(row.K, ST, row.option), axis=1
)

RND.data[cols + ["opt_payoff"]]

a = 1
