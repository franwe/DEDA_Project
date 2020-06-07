import os
import pandas as pd
import pickle

from matplotlib import pyplot as plt

from util.smoothing import locpoly_r, rookley_fixtau
from util.risk_neutral_density import spd_sfe, spd_appfinance, spd_rookley

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN

# ------------------------------------------------------------------- LOAD DATA
d = pd.read_csv(data_path + 'calls_3.csv')
print(d.date.value_counts())
day = '2020-03-29'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 50

for tau_day in df.tau_day.value_counts().index:
    print(tau_day)
    df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]

    # ------------------------------------------------------------------- SMOOTHING
    smoothing_method = locpoly_r
    smoothing_method = rookley_fixtau
    smile, first, second, M, S, K, M_std = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                     gridsize=num, kernel='epak')

    # plt.plot(df_tau.M, df_tau.iv, 'ro', ms=3, alpha=0.3)
    # plt.plot(M, smile)

    # plt.plot(M_std, first)
    # plt.plot(M_std, second)

    # --------------------------------------------------------------- CALCULATE SPD
    r = df.r.iloc[0]
    tau = df_tau.tau.iloc[0]  # TODO: only works because only consider one tau,
                          # no surface

    # has to be M because M_std can be negative, but need log(M)

    for spd in [spd_appfinance]: # [spd_rookley, spd_appfinance, spd_sfe]:
        print(spd)
        result = spd(M, S, K, smile, first, second, r, tau)
        # plt.plot(K, result)

    res.update({tau_day : {'df': df_tau[['M', 'iv', 'S', 'K']],
                'M_std': M_std,
                'M': M,
                'smile': smile,
                'first': first,
                'second': second,
                'K': K,
                'q': result,
                'S': S
                }})

with open(data_path + 'results_' + day + '.pkl', 'wb') as f:
    pickle.dump(res, f)
