import os
from matplotlib import pyplot as plt
from os.path import join
import pandas as pd
import numpy as np

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# --------------------------------------------------------------------- 2D PLOT


def plot_MKM(
    RndData,
    HdData,
    day,
    tau_day,
    x,
    y_lim=None,
    reset_S=False,
    overwrite=False,
    h_densfit=0.2,
    moneyness="K_S",
):
    filename = "T-{}_{}_M-K.png".format(tau_day, day)
    if reset_S:
        filename = "T-{}_{}_M-K_S0.png".format(tau_day, day)

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="complete")
    hd_data, S0 = HdData.filter_data(day)
    print(S0, day, tau_day)
    if reset_S:
        df_tau["S"] = S0

    if moneyness == "K_S":
        print("df_moneyness", moneyness)
        df_tau["M"] = df_tau.K / df_tau.S
    elif moneyness == "S_K":
        df_tau["M"] = df_tau.S / df_tau.K
        print("df_moneyness", moneyness)

    RND = RndCalculator(df_tau, tau_day, day, h_densfit=h_densfit, moneyness=moneyness)
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
        moneyness=moneyness,
    )
    HD.get_hd(variate=True)

    call_mask = RND.data.option == "C"
    RND.data["color"] = "blue"  # blue - put
    RND.data.loc[call_mask, "color"] = "red"  # red - call

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # --------------------------------------------------- Moneyness - Moneyness
    ax = axes[0]
    ax.scatter(RND.data.M, RND.data.q_M, 5, c=RND.data.color)
    ax.plot(RND.M, RND.q_M, "-", c="r")
    ax.plot(HD.M, HD.q_M, "-", c="b")

    ax.text(
        0.99,
        0.99,
        str(day) + "\n" + r"$\tau$ = " + str(tau_day),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_xlim((1 - x), (1 + x))
    if y_lim:
        ax.set_ylim(0, y_lim["M"])
    ax.set_ylim(0)
    ax.vlines(1, 0, RND.data.q_M.max())
    ax.set_xlabel("Moneyness M")

    # ------------------------------------------------------------------ Strike

    ax = axes[1]
    ax.scatter(RND.data.K, RND.data.q, 5, c=RND.data.color)
    ax.plot(RND.K, RND.q_K, "-", c="r")
    ax.plot(HD.K, HD.q_K, "-", c="b")

    ax.text(
        0.99,
        0.99,
        str(day) + "\n" + r"$\tau$ = " + str(tau_day),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_xlim((1 - x) * S0, (1 + x) * S0)
    if y_lim:
        ax.set_ylim(0, y_lim["K"])
    ax.set_ylim(0)
    ax.set_xlabel("Strike Price K")
    ax.vlines(S0, 0, RND.data.q.max())
    plt.tight_layout()
    return fig, filename


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff?


def create_dates(start, end):
    dates = pd.date_range(start, end, closed="right", freq="D")
    return [str(date.date()) for date in dates]


days = create_dates(start="2020-03-01", end="2020-09-30")

for day in days:
    print(day)
    taus = RndData.analyse(day)
    for tau in taus:
        tau_day = tau["_id"]
        if (tau_day > 7) & (tau_day <= 40):  # h_densfit  = 0.15
            # if (tau_day > 40) & (tau_day <= 99):  # h_densfit = 0.25
            try:
                fig, filename = plot_MKM(
                    RndData,
                    HdData,
                    day,
                    tau_day,
                    x=x,
                    reset_S=False,
                    overwrite=False,
                    h_densfit=0.15,
                    moneyness="S_K",
                )
                fig.savefig(join(save_plots, filename), transparent=True)
            except ValueError as e:
                print("ValueError  : ", e, day, tau_day)
            except np.linalg.LinAlgError as e:
                print("np.linalg.LinAlgError :  ", e)
                print("cant invert matrix, smoothing_rookley")
            except ZeroDivisionError as e:
                print("ZeroDivisionError  : ", e)
                print("Empty data.")
