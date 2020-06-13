import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


def BSValue(S, r, sigma, K, T, option):
    d1 = (np.log( S /K) + (r + 0.5*sigma**2 ) *T) /(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == 'Call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == 'Put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(P, S, K, r, T, option, sigma, iterations,
                       convergance_threshold):
    diff_before = 1
    change = 0.1
    i = 0
    while (abs(diff_before) > convergance_threshold) and (i < iterations):

        BS_price = BSValue(S, r, sigma, K, T, option)
        diff_after = P - BS_price  # if negative, need to lower sigma (BS was too high)
        if diff_after > 0:
            sigma *= (1 + change)
        elif diff_after < 0:
            sigma *= (1 - change)

        # if we crossed 0, we change sigma in smaller steps
        if np.sign(diff_before) * np.sign(diff_after) == -1:
            change *= 0.5

        i += 1
        diff_before = diff_after

    # did we stop because of convergance, or because max iterations
    if abs(diff_after) > convergance_threshold:
        print('reached max_iterations: ', i, K, sigma, diff_after)
    # else:
    # print('reached converance threshold after iteration: ', i)

    return (sigma)



# ------------------------------------------------------------------------ MAIN

# ------------------------------------------------------------------- LOAD DATA
d = pd.read_csv(data_path + 'calls_1.csv')
d = d.drop('Unnamed: 0', axis=1)  # TODO: Do this in my script as well?
d = d.drop_duplicates()

# print('exclude values with too big or too smal Moneyness : ',
#       sum(d.M > 1.3) + sum(d.M <= 0.7))
# d = d[d.M <= 1.2]  # filter out Moneyness bigger than 1.3
# d = d[d.M > 0.7]   # filter out Moneyness small than 0.7

print(d.date.value_counts())
day = '2020-03-11'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 50

tau_day = 9 # 16

print(tau_day)
df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
h = df_tau.shape[0] ** (-1 / 9)
tau = df_tau.tau.iloc[0]

# ---------------------------------------------------------------- CALCULATE IV

start_sigma = 0.5
iterations = 500
convergance_threshold = 10**(-9)

df_tau['BS_iv'] = df_tau.apply(lambda row:
                             implied_volatility(P = row.P,
                                                S = row.S,
                                                K = row.K,
                                                r = row.r, T = row.tau, option = 'Call',  # or tau_day?
                                                iterations = iterations,
                                                convergance_threshold = convergance_threshold,
                                                sigma = start_sigma)
                             ,axis=1)


# ------------------------------------------------------------------------ PLOT
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
ax.scatter(df_tau.M, df_tau.iv, c='tab:blue', s=6)
ax.scatter(df_tau.M, df_tau.BS_iv, c='tab:red', s=6)
ax.set_xlabel('Moneyness')
ax.set_ylabel('implied Volatility [%]')
plt.tight_layout()

fig.savefig(data_path + 'ImpliedVola.png', transparent=True)