import os
import pandas as pd
import numpy as np
import pickle

from matplotlib import pyplot as plt

from util.smoothing import locpoly_r, rookley_fixtau
from util.risk_neutral_density import spd_sfe, spd_appfinance, spd_rookley
from util.historical_density import MC_return, SVCJ, density_estimation


cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN

# ---------------------------------------------------------- LOAD DATA ---- RND
d = pd.read_csv(data_path + 'calls_1.csv')
print(d.date.value_counts())
day = '2020-03-11'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 50

tau_day = 9

print(tau_day)
df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
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

# -------------- HD
# ------------------------------------------------------------------- LOAD DATA
data_path = cwd + 'data' + os.sep
d = pd.read_csv(data_path + 'BTCUSDT.csv')

# -------------------------------------------------------------------- SETTINGS
target = 'Adj.Close'
# tau_day = 10
M = 10000
h = 0.1
# day = '2020-03-29'
S0 = d.loc[d.Date == day, target].iloc[0]
# S = np.linspace(S0*0.3, S0*2, num=500)

sample_MC = MC_return(d, target, tau_day, S0, M)
sample_SVCJ, processes = SVCJ(tau_day, S0, M, myseed=1)
sample = sample_SVCJ
S_HD = np.linspace(sample.min() * 0.99, sample.max() * 1.01, num=500)
hd = density_estimation(sample, S_HD, S0, h, kernel='epanechnikov')
# -------------
plt.plot(S, result)
plt.plot(S_HD, hd)