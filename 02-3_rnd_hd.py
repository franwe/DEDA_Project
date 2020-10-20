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
def plot_MKM(RndData, HdData, day, tau_day, x, y_lim=None, reset_S=False, overwrite=False, h_densfit=0.2):
    filename = 'T-{}_{}_M-K.png'.format(tau_day, day)
    if reset_S: filename = 'T-{}_{}_M-K_S0.png'.format(tau_day, day)

    print(day, tau_day)
    hd_data, S0 = HdData.filter_data(day)
    tomorrow = '2020-03-13'
    hd_data, S0 = HdData.filter_data(tomorrow)
    print(S0)
    HD = HdCalculator(hd_data, tau_day = tau_day, date = day,
                      S0=S0, burnin=tau_day*2, path=garch_data, M=5000, overwrite=overwrite)
    HD.get_hd()

    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    if reset_S:
        df_tau['S'] = S0
        df_tau['M'] = df_tau.S / df_tau.K
    RND = RndCalculator(df_tau, tau_day, day, h_densfit=h_densfit)
    RND.fit_smile()
    RND.rookley()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # --------------------------------------------------- Moneyness - Moneyness
    ax = axes[0]
    ax.plot(RND.data.M, RND.data.q_M, '.', markersize=5, color='gray')
    ax.plot(RND.M, RND.q_M, '-', c='r')
    ax.plot(HD.M, HD.q_M, '-', c='b')

    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlim((1-x), (1+x))
    if y_lim: ax.set_ylim(0, y_lim['M'])
    ax.set_ylim(0)
    ax.vlines(1, 0, RND.data.q_M.max())
    ax.set_xlabel('Moneyness M')

    # ------------------------------------------------------------------ Strike

    ax = axes[1]
    ax.plot(RND.data.K, RND.data.q, '.', markersize=5, color='gray')
    ax.plot(RND.K, RND.q_K, '-', c='r')
    ax.plot(HD.K, HD.q_K, '-', c='b')

    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.set_xlim((1-x)*S0, (1+x)*S0)
    if y_lim: ax.set_ylim(0, y_lim['K'])
    ax.set_ylim(0)
    ax.set_xlabel('Strike Price K')
    ax.vlines(S0, 0, RND.data.q.max())
    plt.tight_layout()
    return fig, filename


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=x)
# TODO: Influence of coutoff?

day = '2020-03-12'
RndData.analyse(day)
tau_day = 8
fig, filename = plot_MKM(RndData, HdData, day, tau_day, x=x, reset_S=False, overwrite=False, h_densfit=0.15) #, y_lim={'M': 1.6, 'K': 0.00035})
fig.savefig(join(save_plots, filename), transparent=True)

# ----------------------------------------------------------------------- Plots
files = [f for f in os.listdir(garch_data) if (isfile(join(garch_data, f)) & (f.startswith('T-')) & (f.endswith('.csv')))]
files.sort()

for file in files:
    print(file)
    splits = file.split('_')
    tau_day = int(splits[0][2:])
    day = splits[1]
    fig3, fig4, fig5 = plot_MKM(RndData, HdData, day, tau_day, x=x, reset_S=False)
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
    fig3, fig4, fig5 = plot_MKM(RndData, HdData, day, tau_day, x=x, y_lim=0.00075, reset_S=True)
    figpath = os.path.join(save_plots, 'M-Direct_T-{}_{}.png'.format(tau_day, day))
    fig3.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'M_T-{}_{}.png'.format(tau_day, day))
    fig4.savefig(figpath, transparent=True)
    figpath = os.path.join(save_plots, 'K_T-{}_{}.png'.format(tau_day, day))
    fig5.savefig(figpath, transparent=True)
