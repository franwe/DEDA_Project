import os
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile, join
import matplotlib

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator
from util.density import integrate
from util.data import HdDataClass, RndDataClass

cwd = os.getcwd() + os.sep
source_data = os.path.join(cwd, 'data', '01-processed') + os.sep
save_data = os.path.join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep
garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep


# --------------------------------------------------------------------- 2D PLOT
def plot_MKM(RndData, HdData, day, tau_day, x=0.3, y_lim=None, reset_S=False):
    print(day, tau_day)
    hd_data, S0 = HdData.filter_data(day)
    HD = HdCalculator(hd_data, tau_day = tau_day, date = day,
                      S0=S0, burnin=tau_day*2, path=garch_data, M=1000, overwrite=False)
    HD.get_hd()

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    if reset_S:
        df_tau['S'] = S0
        df_tau['M'] = df_tau.S / df_tau.K
    RND = RndCalculator(df_tau, tau_day, day)
    RND.fit_smile()
    RND.rookley()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    # --------------------------------------------------- Moneyness - Moneyness
    ax = axes[0]
    ax.plot(RND.data.M, RND.data.q, '.', markersize=5, color='gray')
    ax.plot(RND.M, RND.q_fitM, '-', c='r')
    ax.plot(HD.M, HD.q, '-', c='b')

    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlim((1-x), (1+x))
    if y_lim: ax.set_ylim(0, y_lim)
    ax.set_xlabel('Moneyness M')

    # ------------------------------------------------------ Strike - Moneyness
    RNDdataK_inv = (RND.data.K/S0)**(-1)
    RNDK_inv = (RND.K/S0)**(-1)

    ax = axes[1]
    ax.plot(RNDdataK_inv, RND.data.q, '.', markersize=5, color='gray')
    ax.plot(RNDK_inv, RND.q_fitK, '-', c='r')
    ax.plot(HD.M, HD.q, '-', c='b')

    # ax.plot(RND.data.M, RND.data.q, '.', markersize=5, color='gray')
    # ax.plot(RND.M, RND.q_fitM, '-', c='r')
    # ax.plot(HD.M, HD.q_S, '-', c='b')

    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.set_xlim(1-x, 1+x)
    if y_lim: ax.set_ylim(0, y_lim)
    ax.set_xlabel('Moneyness M')

    # ------------------------------------------------------------------ Strike

    ax = axes[2]
    ax.plot(RND.data.K, RND.data.q, '.', markersize=5, color='gray')
    ax.plot(RND.K, RND.q_fitK, '-', c='r')
    ax.plot(HD.K, HD.q, '-', c='b')

    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.set_xlim((1-x)*S0, (1+x)*S0)
    if y_lim: ax.set_ylim(0, y_lim)
    ax.set_xlabel('Strike Price K')

    plt.tight_layout()

    return fig


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=x)
# TODO: Influence of coutoff?

# ----------------------------------------------------------------------- Plots
files = [f for f in os.listdir(garch_data) if (isfile(join(garch_data, f)) & (f.startswith('T-')) & (f.endswith('.csv')))]
files.sort()

for file in files:
    print(file)
    splits = file.split('_')
    tau_day = int(splits[0][2:])
    day = splits[1]
    fig3, fig4, fig5 = plot_2d(RndData, HdData, day, tau_day, x=x, reset_S=False)
    figpath = os.path.join(save_plots, 'M-Direct_T-{}_{}.png'.format(tau_day, day))
    fig3.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'M_T-{}_{}.png'.format(tau_day, day))
    fig4.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'K_T-{}_{}.png'.format(tau_day, day))
    fig5.savefig(figpath, transparent=True)


# ----------------------------------------------------------------------- Plots

# ----------------------------------------------------------------------- Plots
files = [f for f in os.listdir(garch_data) if (isfile(join(garch_data, f)) & (f.startswith('T-7')) & (f.endswith('.csv')))]
files.sort()

for file in files:
    print(file)
    splits = file.split('_')
    tau_day = int(splits[0][2:])
    day = splits[1]
    fig3, fig4, fig5 = plot_2d(RndData, HdData, day, tau_day, x=x, y_lim=0.00075, reset_S=True)
    figpath = os.path.join(save_plots, 'M-Direct_T-{}_{}.png'.format(tau_day, day))
    fig3.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'M_T-{}_{}.png'.format(tau_day, day))
    fig4.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'K_T-{}_{}.png'.format(tau_day, day))
    fig5.savefig(figpath, transparent=True)

fig= plot_MKM(RndData, HdData, day, tau_day, x=0.3, y_lim=None, reset_S=False)

fig4, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0] = ax1
