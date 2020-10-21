import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from util.data import RndDataClass

cwd = os.getcwd() + os.sep
data_path = cwd + "data" + os.sep


def BSValue(S, r, sigma, K, T, option):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(P, S, K, r, T, option, sigma, iterations, convergance_threshold):
    diff_before = 1
    change = 0.1
    i = 0
    while (abs(diff_before) > convergance_threshold) and (i < iterations):

        BS_price = BSValue(S, r, sigma, K, T, option)
        # if negative, need to lower sigma (BS was too high)
        diff_after = P - BS_price
        if diff_after > 0:
            sigma *= 1 + change
        elif diff_after < 0:
            sigma *= 1 - change

        # if we crossed 0, we change sigma in smaller steps
        if np.sign(diff_before) * np.sign(diff_after) == -1:
            change *= 0.5

        i += 1
        diff_before = diff_after

    # did we stop because of convergance, or because max iterations
    if abs(diff_after) > convergance_threshold:
        print("reached max_iterations: ", i, K, sigma, diff_after)

    return sigma


def calculate_iv(
    df_tau, start_sigma=0.5, iterations=500, convergance_threshold=10 ** (-9)
):
    calls = df_tau[df_tau.option == "C"]
    puts = df_tau[df_tau.option == "P"]

    calls["BS_iv"] = calls.apply(
        lambda row: implied_volatility(
            P=row.P,
            S=row.S,
            K=row.K,
            r=row.r,
            T=row.tau,
            option="Call",
            iterations=iterations,
            convergance_threshold=convergance_threshold,
            sigma=start_sigma,
        ),
        axis=1,
    )

    puts["BS_iv"] = puts.apply(
        lambda row: implied_volatility(
            P=row.P,
            S=row.S,
            K=row.K,
            r=row.r,
            T=row.tau,
            option="Put",
            iterations=iterations,
            convergance_threshold=convergance_threshold,
            sigma=start_sigma,
        ),
        axis=1,
    )

    full = pd.concat([calls, puts], axis=1)
    return full


# ------------------------------------------------------------------------ MAIN
# ------------------------------------------------------------------- LOAD DATA
d = RndData.complete

print(d.date.value_counts())
day = "2020-03-11"
df = d[(d.date == day)]
print(df.tau_day.value_counts())
tau_day = 9

df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
print(
    "Calculate IV for {} options, on {} with maturity T={}.".format(
        df_tau.shape[0], day, tau_day
    )
)

RndData = RndDataClass(cutoff=0.5)
df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="complete")
# ---------------------------------------------------------------- CALCULATE IV
full = calculate_iv(df_tau)

# ------------------------------------------------------------------------ PLOT
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
ax.scatter(full.M, full.iv, c="tab:blue", s=6)
ax.scatter(full.M, full.BS_iv, c="tab:red", s=6)
ax.set_xlabel("Moneyness")
ax.set_ylabel("implied Volatility [%]")
plt.tight_layout()

figpath = join(data_path, "plots", "ImpliedVola_{}_T{}.png".format(day, tau_day))
fig.savefig(figpath, transparent=True)
