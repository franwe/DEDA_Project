import os
from matplotlib import pyplot as plt
# from localreg import *
import numpy as np

from util.data import RndDataClass, HdDataClass
from util.smoothing_f import local_polynomial
from util.smoothing import bspline
from util.risk_neutral_density_bu import spd_appfinance
from util.historical_density import get_hd

cwd = os.getcwd() + os.sep
source_data = os.path.join(cwd, 'data', '01-processed') + os.sep
save_data = os.path.join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep


# --------------------------------------------------------------------- 2D PLOT
def plot_2d(RndData, HdData, day, tau_day, x=0.3, y_lim=None):

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0

    fig3 = plt.figure(figsize=(5,4))
    ax3 = fig3.add_subplot(111)
    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_appfinance
    X = np.array(df_tau.M)
    Y = np.array(df_tau.iv)
    smile, first, second, M, f = local_polynomial(X, Y, h)

    # ---------------------------------------- B-SPLINE on SMILE, FIRST, SECOND
    pars, spline, points = bspline(M, smile, sections=8, degree=3)
    # derivatives
    first_fct = spline.derivative(1)
    second_fct = spline.derivative(2)

    df_tau['q'] = df_tau.apply(lambda row: spd(row.M, row.S, row.K,
                                               spline(row.M), first_fct(row.M),
                                               second_fct(row.M),
                                               r, tau), axis=1)

    Q = np.array(df_tau.q)
    ax3.plot(X, Q, '.', markersize=5, color='gray')

    fit, first, second, X_domain, f = local_polynomial(X, Q, h=0.1*np.mean(X), kernel='epak')
    ax3.plot(X_domain, fit, '-', c='r')

    # ---------------------------------------------------------------------- HD
    hd, M = get_hd(HdData, day, tau_day)

    ax3.plot(M, hd, '-', c='b')

    ax3.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax3.transAxes)
    # ax3.axvline(1, ls=':')
    ax3.set_xlim(1-x, 1+x)
    if y_lim: ax3.set_ylim(0, y_lim)
    ax3.set_xlabel('Moneyness')
    plt.tight_layout()
    return fig3


# -----------------------------------------------------------------------------
day = '2020-03-11'
tau_day = 9
x = 0.3

# ----------------------------------------------------------- LOAD DATA HD, RND
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=x)
# TODO: Influence of coutoff?

# ----------------------------------------------------------------------- Plots
days = ['2020-03-11', '2020-03-20', '2020-03-29', '2020-03-06']
taus = [2,             2,            2,            21]

days = ['2020-03-07', '2020-03-11', '2020-03-18', '2020-03-23', '2020-03-30', '2020-04-04']
taus = [2, 2,2,2,2,2]

days = ['2020-03-06', '2020-03-13', '2020-03-20', '2020-04-03']
taus = [14,             14,            14,            14]


days = ['2020-03-06', '2020-03-13', '2020-03-06']
taus = [14,             14,            7]

for day, tau_day in zip(days, taus):
    print(day)
    fig3 = plot_2d(RndData, HdData, day, tau_day, x=x)
    figpath = os.path.join(data_path, 'plots', 'T-{}_{}.png'.format(tau_day, day))
    fig3.savefig(figpath, transparent=True)



RndData.analyse(sortby='date')
day = '2020-03-11'
RndData.analyse(day)
tau_day = 9

df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
hd_data, S0 = HdData.filter_data(date=day)
fig3 = plot_2d(df_tau, day, tau_day, hd_data, S0)
figpath = os.path.join(data_path, 'plots', 'T-{}_{}.png'.format(tau_day, day))
fig3.savefig(figpath, transparent=True)
