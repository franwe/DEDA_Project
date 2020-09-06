import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from util.smoothing import local_polynomial, bspline
from util.risk_neutral_density_bu import spd_sfe, spd_appfinance, spd_rookley
from util.expand import expand_X
from util.historical_density import sampling, density_estimation
# from util.garch import simulate_hd

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN



def something_3d_plot():
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
        smoothing_method = local_polynomial
        smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                         gridsize=num, kernel='epak')
        y_3d = [i]*len(K)
        result = spd(M, S, K, smile, first, second, r, tau)
        ax1.plot(K[::-1],  y_3d, result, c=c)
        ax3.plot(K[::-1], result, c=c)


        # -------------------------------------------------------------------- B-SPLINE
        M_long, M_left, M_right = expand_X(M, add_left=70, add_right=70)
        S_long, S_left, S_right = expand_X(S, add_left=70, add_right=70)
        K_long, K_left, K_right = expand_X(K, add_left=70, add_right=70)
        y_3d_long = [i] * len(K_long)

        pars, spline, points = bspline(M, smile, sections=8, degree=3)

        # derivatives
        first_fct = spline.derivative(1)
        second_fct = spline.derivative(2)

        smile_long = spline(M_long)
        first_long = first_fct(M_long)
        second_long = second_fct(M_long)

        result_long = spd(M_long, S_long, K_long, smile_long, first_long, second_long, r, tau)
        ax1.plot(K_long[::-1], y_3d_long, result_long, ':', c=c)
        ax3.plot(K_long[::-1], result_long, ':', c=c)

        # ---------------------------------------------------------------------- HD
        sample_MC = sampling(d_usd, target, tau_day, S0, n)
        sample = sample_MC
        S_hd = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
        y_3d_hd = [i]*len(S_hd)
        hd = density_estimation(sample, S_hd, S0*h_hd, kernel='epanechnikov')
        ax1.plot(S_hd, y_3d_hd, hd, '-', c=c)
        ax3.plot(S_hd, hd, '-', c=c)
        return fig1

# something_3d_plot()
# --------------------------------------------------------------------- 2D PLOT

def plot_2d(d, day, tau_day):
    sum(d.date==day)
    d = d.reset_index()
    df_tau = d[(d.tau_day == tau_day) & (d.date == day)]
    df_tau = df_tau.reset_index()
    df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0


    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_appfinance
    smoothing_method = local_polynomial
    smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                 gridsize=num, kernel='epak')
    result = spd(M, S, K, smile, first, second, r, tau)
    ax3.plot(K[::-1], result, c='r')


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
    ax3.plot(K_long[::-1], result_long, ':', c='r')

    # ---------------------------------------------------------------------- HD
    # sample = sampling(d_usd, target, tau_day, S0, n)
    # S_hd = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
    # hd = density_estimation(sample, S_hd, S0*h_hd, kernel='epanechnikov')
    # ax3.plot(S_hd, hd, '-', c='b')
    hd, S_hd, S0 = simulate_hd(d_usd, day, tau_day, target)
    ax3.plot(S_hd, hd, '-', c='b')

    ax3.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax3.transAxes)

    return fig3, df_tau


# --------------------------------------------------------------- LOAD DATA RND
d = pd.read_csv(data_path + 'trades_clean.csv')
print('exclude values with too big or too smal Moneyness : ',
      sum(d.M > 1.2) +  sum(d.M <= 0.7))
d = d.drop('Unnamed: 0', axis=1)
d = d[d.M <= 1.2]  # filter out Moneyness bigger than 1.2
d = d[d.M > 0.7]   # filter out Moneyness small than 0.7

print(d.date.value_counts())
day = '2020-03-11'
df = d[(d.date == day)]
print(df.tau_day.value_counts())
res = dict()
num = 140

# ---------------------------------------------------------------- LOAD DATA HD
data_path = cwd + 'data' + os.sep
d_usd = pd.read_csv(data_path + 'BTCUSDT.csv')
target = 'Adj.Close'
n = 10000
h_hd = 0.11
S0 = d_usd.loc[d_usd.Date == day, target].iloc[0]

tau_day = 2

day = '2020-03-11'
fig3 = plot_2d(d, day, tau_day)

day = '2020-03-20'
fig3 = plot_2d(d, day, tau_day)

day = '2020-03-29'
fig3 = plot_2d(d, day, tau_day)


# -------------------------------------------------------------------------------------
day = '2020-03-06'  # TODO: INTERESTING! - Check Domain for S HD: to short, maybe thats why too high
tau_day = 21
fig3, old = plot_2d(d, day, tau_day)

