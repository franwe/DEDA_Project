import os
import pandas as pd
import numpy as np
import pickle
import math

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from util.smoothing import locpoly_r, rookley_fixtau, bspline
from util.risk_neutral_density import spd_sfe, spd_appfinance, spd_rookley
from util.expand import expand_X
from util.historical_density import MC_sample, density_estimation

import imageio

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN

# ---------------------------------------------------------- LOAD DATA ---- RND
# d = pd.read_csv(data_path + 'calls_1.csv')
d = pd.read_csv(data_path + 'trades_clean.csv')
d = d.drop('Unnamed: 0', axis=1)
print('exclude values with too big or too small Moneyness : ',
      sum(d.M > 1.3) + sum(d.M <= 0.7))
d = d[d.M <= 1.2]  # filter out Moneyness bigger than 1.3
d = d[d.M > 0.7]   # filter out Moneyness small than 0.7

print(d.date.value_counts())
day = '2020-03-11'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 140
tau_day = 2

a = d.groupby(by=['tau_day', 'date']).count()
a.to_csv(data_path + 'taus.csv')

# ---------------------------------------------------------------- LOAD DATA HD
data_path = cwd + 'data' + os.sep
d_usd = pd.read_csv(data_path + 'BTCUSDT.csv')
target = 'Adj.Close'
n = 10000
h_hd = 0.11
S0 = d_usd.loc[d_usd.Date == day, target].iloc[0]


from mpl_toolkits.mplot3d import Axes3D
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111, projection='3d')
taus = [1, 2, 9, 16, 44, 107, 198]
color=cm.rainbow(np.linspace(0,1,len(taus)))

for tau_day, c, i in zip(taus, color, range(1, len(taus)+1)):
    print(tau_day)
    df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
    df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0

    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_sfe
    smoothing_method = locpoly_r
    smoothing_method = rookley_fixtau
    smile, first, second, M, S, K, M_std = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                     gridsize=num, kernel='epak')
    y_3d = [i]*len(K)
    result = spd(M, S, K, smile, first, second, r, tau)
    ax1.plot(K[::-1],  y_3d, result, c=c)
    ax3.plot(K[::-1], result, c=c)


    # -------------------------------------------------------------------- B-SPLINE

    pars, spline, points = bspline(M, smile, sections=8, degree=3)

day = '2020-03-09'
def density_plot(d, day, tau_day):
    df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
    df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0

    S0 = d_usd.loc[d_usd.Date == day, target].iloc[0]
    fig2 = plt.figure(figsize=(4, 4))
    ax = fig2.add_subplot(111)
    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_sfe
    smoothing_method = rookley_fixtau
    smile, first, second, M, S, K, M_std = smoothing_method(df_tau, tau, h,
                                                            h_t=0.1,
                                                            gridsize=num,
                                                            kernel='epak')
    result = spd(M, S, K, smile, first, second, r, tau)
    ax.plot(K[::-1], result, c=c)

    # -------------------------------------------------------------------- B-SPLINE
    M_long, M_left, M_right = expand_X(M, add_left=70, add_right=70)
    S_long, S_left, S_right = expand_X(S, add_left=70, add_right=70)
    K_long, K_left, K_right = expand_X(K, add_left=70, add_right=70)

    pars, spline, points = bspline(M, smile, sections=8, degree=3)
    # derivatives
    first_fct = spline.derivative(1)
    second_fct = spline.derivative(2)

    smile_long = spline(M_long)
    first_long = first_fct(M_long)
    second_long = second_fct(M_long)

    result_long = spd(M_long, S_long, K_long, smile_long, first_long, second_long, r, tau)
    ax.plot(K_long[::-1], result_long, ':', c=c)

    ax.text(0.99, 0.99, str(day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlabel('spot price')
    ax.set_xlim(S0*0.3, S0*1.7)
    ax.set_ylim(0, 0.0015)
    plt.tight_layout()

    # Used to return the plot as an image rray
    fig2.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    return image

for day in list(d.date.value_counts().index):
    density_plot(d, day, tau_day=2)


kwargs_write = {'fps': 5.0, 'quantizer': 'nq'}
imageio.mimsave(data_path + 'RND_tau2.gif',
                [density_plot(d, day, tau_day=2)
                 for day in list(d[d.tau_day == 2].date.value_counts().index)], fps=5)