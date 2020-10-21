import os
import pickle
from matplotlib import pyplot as plt

cwd = os.getcwd() + os.sep
data_path = cwd + "data" + os.sep

day = "2020-03-11"
res = pickle.load(open(data_path + "results_" + day + ".pkl", "rb"))

# ------------------------------------------------------------------ GRID PLOTS
fig1, axes = plt.subplots(2, 4, figsize=(10, 7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]["df"].M, res[key]["df"].iv, ".")
    ax.plot(res[key]["M"], res[key]["smile"])
    ax.text(
        0.99,
        0.99,
        r"$\tau$ = " + str(key),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
axes.flatten()[0].set_ylabel("implied volatility")
axes.flatten()[4].set_ylabel("implied volatility")
axes.flatten()[4].set_xlabel("moneyness")
axes.flatten()[5].set_xlabel("moneyness")
axes.flatten()[6].set_xlabel("moneyness")
axes.flatten()[7].set_xlabel("moneyness")
plt.tight_layout()
figpath = join(data_path, "plots", "{}_smiles.png".format(day))
fig1.savefig(figpath, transparent=True)


fig2, axes = plt.subplots(2, 4, figsize=(10, 7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]["M_df"], res[key]["q"], ".", markersize=2)
    ax.plot(res[key]["M_df"], res[key]["y2"])
    ax.text(
        0.99,
        0.99,
        r"$\tau$ = " + str(key),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_yticks([])
axes.flatten()[0].set_ylabel("risk neutral density")
axes.flatten()[4].set_ylabel("risk neutral density")
axes.flatten()[4].set_xlabel("BTC price")
axes.flatten()[5].set_xlabel("BTC price")
axes.flatten()[6].set_xlabel("BTC price")
axes.flatten()[7].set_xlabel("BTC price")
plt.tight_layout()
figpath = join(data_path, "plots", "{}_rnds.png".format(day))
fig2.savefig(figpath, transparent=True)


fig3, axes = plt.subplots(2, 4, figsize=(10, 7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]["M"], res[key]["smile"])
    ax.plot(res[key]["M"], res[key]["first"])
    ax.plot(res[key]["M"], res[key]["second"])
    ax.text(
        0.99,
        0.01,
        r"$\tau$ = " + str(key),
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_yticks([])
axes.flatten()[0].set_ylabel("implied volatility")
axes.flatten()[4].set_ylabel("implied volatility")
axes.flatten()[4].set_xlabel("moneyness")
axes.flatten()[5].set_xlabel("moneyness")
axes.flatten()[6].set_xlabel("moneyness")
axes.flatten()[7].set_xlabel("moneyness")
plt.tight_layout()
figpath = join(data_path, "plots", "{}_derivatives.png".format(day))
fig3.savefig(figpath, transparent=True)


# ----------------------------------------------------------------- TAU PROCESS
for key in res:
    s = res[key]

    fig4, axes = plt.subplots(1, 3, figsize=(10, 4))
    ax = axes[0]
    ax.plot(s["df"].M, s["df"].iv, ".", c="r")
    ax.plot(s["M"], s["smile"])
    ax.set_xlabel("moneyness")
    ax.set_ylabel("implied volatility")

    ax = axes[1]
    ax.plot(s["M"], s["smile"])
    ax.plot(s["M"], s["first"])
    ax.plot(s["M"], s["second"])
    ax.set_xlabel("moneyness")
    ax.set_ylabel("implied volatility")

    ax = axes[2]
    ax.plot(s["M_df"], s["y2"])
    ax.set_xlabel("BTC price")
    ax.set_ylabel(r"risk neutral density")
    ax.set_yticks([])

    plt.tight_layout()
    figpath = join(data_path, "plots", "T-{}_{}_rnd-hist.png".format(key, day))
    fig4.savefig(figpath, transparent=True)


# --------------------------------------------------------------- GARCH EXPLAIN
# load everything in garch first
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

from util.data import HdDataClass

HdData = HdDataClass(data_path + "BTCUSDT.csv")
target = "Adj.Close"
days = [
    "2020-03-06",
    "2020-03-11",
    "2020-03-14",
    "2020-03-18",
    "2020-03-20",
    "2020-03-29",
]
days = [
    "2020-03-07",
    "2020-03-11",
    "2020-03-18",
    "2020-03-23",
    "2020-03-30",
    "2020-04-04",
]  # 2
days = ["2020-03-07", "2020-03-14", "2020-03-21", "2020-04-04"]  # 6
days = ["2020-03-06", "2020-03-13", "2020-03-20", "2020-04-03"]  # 14
tau_day = 14
colors = cm.rainbow(np.linspace(0, 1, len(days)))
fig3, axes = plt.subplots(1, 2, figsize=(10, 4))

y_pos = 0.99
x_pos = 0.25
for day, c in zip(days[::-1], colors):
    data, S0 = HdData.filter_data(date=day)
    returns = get_returns(data, target, mode="log")

    res = fit_sigma(returns)
    print(res.x)
    w, a, b = res.x

    sigma2 = _ARMA(w, a, b, returns)  # for start values, use model until now

    # Simulation
    mu = np.mean(sigma2)
    sigma2_0 = sigma2[-1]
    ret_0 = returns.iloc[-1] + mu
    print("mu: {}, sigma0: {}, ret0: {}, S0: {}".format(mu, sigma2_0, ret_0, S0))

    sigma2 = pd.Series(sigma2)

    axes[0].plot(sigma2[0:], c=c)
    axes[0].plot(sigma2[-1:], ".", c=c, ms=15)
    axes[1].plot(returns[0:], c=c)
    axes[1].plot(returns[-1:], ".", c=c, ms=15)

    axes[0].text(
        x_pos,
        y_pos,
        str(day),
        horizontalalignment="right",
        verticalalignment="top",
        transform=axes[0].transAxes,
        c=c,
    )
    y_pos -= 0.05
axes[0].set_xlabel("time index")
axes[0].set_ylabel(r"$\sigma^2_t$")
axes[0].set_title(r"$\sigma^2_t$")

axes[1].set_xlabel("time index")
axes[1].set_ylabel(r"$r_t$")
axes[1].set_title(r"$r_t$")

plt.tight_layout()

figpath = join(data_path, "plots", "T-{}_GARCH.png".format(tau_day))
fig3.savefig(figpath, transparent=True)
