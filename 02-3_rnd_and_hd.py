import os
import pandas as pd
import numpy as np
import pickle
import math

from matplotlib import pyplot as plt

from util.smoothing import locpoly_r, rookley_fixtau
from util.risk_neutral_density import spd_sfe, spd_appfinance, spd_rookley
from util.expand import expand
from util.historical_density import MC_return, SVCJ, density_estimation


cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN

# ---------------------------------------------------------- LOAD DATA ---- RND
d = pd.read_csv(data_path + 'calls_1.csv')
d = pd.read_csv(data_path + 'trades_clean.csv')
d = d.drop('Unnamed: 0', axis=1)  # TODO: Do this in my script as well?
d = d.drop_duplicates()
print('exclude values with too big or too smal Moneyness : ',
      sum(d.M > 1.3) + sum(d.M <= 0.7))
d = d[d.M <= 1.2]  # filter out Moneyness bigger than 1.3
d = d[d.M > 0.7]   # filter out Moneyness small than 0.7

print(d.date.value_counts())
day = '2020-03-11'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 50

tau_day = 9

print(tau_day)
df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
h = df_tau.shape[0] ** (-1 / 9)
tau = df_tau.tau.iloc[0]

# ------------------------------------------------------------------- SMOOTHING
smoothing_method = locpoly_r
smoothing_method = rookley_fixtau
smile, first, second, M, S, K, M_std = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                 gridsize=num, kernel='epak')


# --------------------------------------------------------------- CALCULATE SPD
r = df.r.iloc[0]
tau = df_tau.tau.iloc[0]  # TODO: only works because only consider one tau,
                      # no surface

# has to be M because M_std can be negative, but need log(M)
# for spd in [spd_appfinance]: # [spd_rookley, spd_appfinance, spd_sfe]:

fig2 = plt.figure(figsize=(7, 5))
ax = fig2.add_subplot(111)
# for spd in [spd_rookley, spd_appfinance, spd_sfe]:
for spd in [spd_sfe]: # [spd_rookley, spd_appfinance, spd_sfe]:

    result = spd(M, S, K, smile, first, second, r, tau)
    ax.plot(K[::-1], result)

    # TODO: doesnt work anymore :-(  --- solution in bspline, need to have same add_left and add_right!
    # # expand
    # exp = expand(smile, first, second, M, S, K, edge=0.4)
    # smile_long, first_long, second_long, M_long, S_long, K_long = exp
    # result_long = spd(M_long, S_long, K_long,
    #                   smile_long, first_long, second_long, r, tau)
    # ax.plot(K_long[::-1], result_long, c='b', ls=':', label=str(spd))


# -------------- HD
# ------------------------------------------------------------------- LOAD DATA
data_path = cwd + 'data' + os.sep
d_usd = pd.read_csv(data_path + 'BTCUSDT.csv')

# -------------------------------------------------------------------- SETTINGS
target = 'Adj.Close'
# tau_day = 10
n = 10000
h = 0.1
# day = '2020-03-29'
S0 = d_usd.loc[d_usd.Date == day, target].iloc[0]
# S = np.linspace(S0*0.3, S0*2, num=500)

sample_MC = MC_return(d_usd, target, tau_day, S0, n)
sample_SVCJ, processes = SVCJ(tau_day, S0, n, myseed=1)

# fig2 = plt.figure(figsize=(10, 6))
# ax = fig2.add_subplot(111)

# for name, sample in zip(['MC', 'SVCJ'], [sample_MC, sample_SVCJ]):
for name, sample in zip(['MC'], [sample_MC]):
    S_hd = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
    hd = density_estimation(sample, S_hd, S0, h, kernel='epanechnikov')
    ax.plot(S_hd, hd, '-', c='r', label=name)

ax.set_xlim(S0*0.7, S0*1.3)
# ax.set_ylim(0, 0.0005)
ax.set_xlabel('spot price')
plt.tight_layout()

fig2.savefig(data_path + day + '_RND_HD.png', transparent=True)
