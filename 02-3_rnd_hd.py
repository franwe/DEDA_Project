import os
from matplotlib import pyplot as plt
import numpy as np

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator, integrate

cwd = os.getcwd() + os.sep
source_data = os.path.join(cwd, 'data', '01-processed') + os.sep
save_data = os.path.join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep
garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep


# --------------------------------------------------------------------- 2D PLOT
def plot_2d(RndData, HdData, day, tau_day, x=0.3, y_lim=None):

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    RND = RndCalculator(df_tau, tau_day, day)
    RND.fit_smile()
    RND.rookley()

    hd_data, S0 = HdData.filter_data(day)
    HD = HdCalculator(hd_data, tau_day, day, S0, garch_data)
    HD.get_hd()

    # ------------------------------------------------------------- Spot/Strike
    integrate(RND.K, RND.q_fitK)
    integrate(HD.S, HD.q_S)

    fig3 = plt.figure(figsize=(5,4))
    ax3 = fig3.add_subplot(111)
    ax3.plot(RND.data.K, RND.data.q, '.', markersize=5, color='gray')
    ax3.plot(RND.K, RND.q_fitK, '-', c='r')
    ax3.plot(HD.S, HD.q_S, '-', c='b')

    ax3.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax3.transAxes)
    # ax3.axvline(1, ls=':')
    ax3.set_xlim((1-x)*S0, (1+x)*S0)
    # if y_lim: ax3.set_ylim(0, y_lim)
    ax3.set_xlabel('S')
    plt.tight_layout()

    # --------------------------------------------------------------- Moneyness
    integrate(RND.M, RND.q_fitM)
    integrate(HD.M, HD.q_S)

    fig4 = plt.figure(figsize=(5, 4))
    ax4 = fig4.add_subplot(111)
    ax4.plot(RND.data.M, RND.data.q, '.', markersize=5, color='gray')
    ax4.plot(RND.M, RND.q_fitM, '-', c='r')
    ax4.plot(HD.M, HD.q_S, '-', c='b')

    ax4.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax4.transAxes)
    # ax3.axvline(1, ls=':')
    ax4.set_xlim(1-x, 1+x)
    # if y_lim: ax3.set_ylim(0, y_lim)
    ax4.set_xlabel('Moneyness')
    plt.tight_layout()

    return fig3, fig4


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
taus = [2,2,2,2,2,2]

days = ['2020-03-06', '2020-03-13', '2020-03-20', '2020-04-03']
taus = [14,             14,            14,            14]


days = ['2020-03-06', '2020-03-13', '2020-03-06']
taus = [14,             14,            7]

days = ['2020-03-07', '2020-03-11', '2020-03-18', '2020-03-23', '2020-03-30', '2020-04-04']
taus = [2,2,2,2,2,2]

for day, tau_day in zip(days, taus):
    print(day)
    fig3, fig4 = plot_2d(RndData, HdData, day, tau_day, x=x)
    figpath = os.path.join(save_plots, 'M_T-{}_{}.png'.format(tau_day, day))
    fig3.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'S_T-{}_{}.png'.format(tau_day, day))
    fig4.savefig(figpath, transparent=True)
