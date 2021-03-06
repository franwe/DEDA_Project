import os
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import pickle

from util.general import create_dates, load_tau_section_parameters
from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator, plot_rookleyMethod
from util.historical_density import HdCalculator
from util.density import hd_rnd_domain

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# --------------------------------------------------------------------- 2D PLOT


def save_data_to_pickle(day, tau_day, RND, rnd, hd, M, K, fig):
    data = {
        "date": day,
        "tau_day": tau_day,
        "RND": RND,
        "rnd": rnd,
        "hd": hd,
        "M": M,
        "kernel": K,
        "figure": fig,
    }
    with open(save_data + "T-{}_{}.pkl".format(tau_day, day), "wb") as handle:
        pickle.dump(data, handle)


def plot_MKM(
    RndData,
    HdData,
    day,
    tau_day,
    x,
    y_lim=None,
    reset_S=True,
    overwrite=False,
    h_densfit=0.2,
    save=False,
):
    filename = "T-{}_{}_M-K.png".format(tau_day, day)
    if reset_S:
        filename = "T-{}_{}_M-K_S0.png".format(tau_day, day)

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="complete")
    hd_data, S0 = HdData.filter_data(day)
    print(S0, day, tau_day)
    if reset_S:
        df_tau["S"] = S0
        df_tau["M"] = df_tau.S / df_tau.K

    RND = RndCalculator(df_tau, tau_day, day)
    RND.fit_smile()
    RND.rookley()

    # algorithm plot
    fig_method = plot_rookleyMethod(RND)
    plt.tight_layout()
    # plt.show()
    fig_method.savefig(
        join(save_plots, "RND", filename),
        transparent=True,
    )

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

    # ------------------------------------------------- Kernel K = q/p = rnd/hd
    hd, rnd, M = hd_rnd_domain(
        HD, RND, interval=[RND.data.M.min() * 0.99, RND.data.M.max() * 1.01]
    )
    K = rnd / hd
    ax = axes[1]
    ax.plot(M, K, "-", c="k")
    ax.axhspan(0.7, 1.3, color="grey", alpha=0.5)
    ax.text(
        0.99,
        0.99,
        str(day) + "\n" + r"$\tau$ = " + str(tau_day),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_xlim((1 - x), (1 + x))
    ax.set_ylim(0, 2)
    ax.set_ylabel("K = rnd / hd")
    ax.set_xlabel("Moneyness")
    plt.tight_layout()

    if save:
        save_data_to_pickle(day, tau_day, RND, rnd, hd, M, K, fig)

    return fig, filename


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff?

(_, _, _, h, tau_min, tau_max) = load_tau_section_parameters("huge")

days = create_dates(start="2020-03-01", end="2020-09-30")
for day in days:
    print(day)
    taus = RndData.analyse(day)
    for tau in taus:
        tau_day = tau["_id"]
        if (tau_day > tau_min) & (tau_day <= tau_max):
            try:
                fig, filename = plot_MKM(
                    RndData,
                    HdData,
                    day,
                    tau_day,
                    x=x,
                    reset_S=False,
                    overwrite=False,
                    h_densfit=h,
                    save=True,
                )
                fig.savefig(
                    join(save_plots, "kernel", filename), transparent=True
                )
            except ValueError as e:
                print("ValueError  : ", e, day, tau_day)
            except np.linalg.LinAlgError as e:
                print("np.linalg.LinAlgError :  ", e)
                print("cant invert matrix, smoothing_rookley")
            except ZeroDivisionError as e:
                print("ZeroDivisionError  : ", e)
                print("Empty data.")
