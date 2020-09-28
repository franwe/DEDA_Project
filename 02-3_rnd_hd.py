import os
from matplotlib import pyplot as plt
from localreg import *

from util.data import RndDataClass, HdDataClass
from util.smoothing_f import local_polynomial
from util.smoothing import bspline
from util.risk_neutral_density_bu import spd_appfinance
from util.garch import simulate_hd

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


# --------------------------------------------------------------------- 2D PLOT
def plot_2d(df_tau, day, tau_day, hd_data, S0, x=0.3, y_lim=None):
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0

    fig3 = plt.figure(figsize=(5,4))
    ax3 = fig3.add_subplot(111)
    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_appfinance
    smoothing_method = local_polynomial
    X = np.array(df_tau.M)
    Y = np.array(df_tau.iv)
    smile, first, second, M, f = smoothing_method(X, Y, h)

    # ---------------------------------------- B-SPLINE on SMILE, FIRST, SECOND
    pars, spline, points = bspline(M, smile, sections=8, degree=3)
    # derivatives
    first_fct = spline.derivative(1)
    second_fct = spline.derivative(2)

    df_tau['q'] = df_tau.apply(lambda row: spd(row.M, row.S, row.K,
                                               spline(row.M), first_fct(row.M),
                                               second_fct(row.M),
                                               r, tau), axis=1)

    a = df_tau.sort_values('M')
    M_df = a.M.values
    q_df = a.q.values
    ax3.plot(M_df, q_df, '.', markersize=5, color='gray')

    fit, first, second, X_domain, f = local_polynomial(M_df, q_df, h=0.1, kernel='epak')
    ax3.plot(X_domain, fit, '-', c='r')

    # ---------------------------------------------------------------------- HD
    S_hd = np.linspace(M_df.min()*S0, M_df.max()*S0, num=100)
    hd, S_hd = simulate_hd(hd_data, S0, tau_day, S_domain=S_hd)
    ax3.plot(S_hd/S0, hd, '-', c='b')

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
HdData = HdDataClass(data_path + 'BTCUSDT.csv')
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
# TODO: Influence of coutoff?

# ----------------------------------------------------------------------- Plots
days = ['2020-03-11', '2020-03-20', '2020-03-29', '2020-03-06']
taus = [2,             2,            2,            21]

days = ['2020-03-07', '2020-03-11', '2020-03-18', '2020-03-23', '2020-03-30', '2020-04-04']
taus = [2, 2,2,2,2,2]

days = ['2020-03-06', '2020-03-13', '2020-03-20', '2020-04-03']
taus = [14,             14,            14,            14]

for day, tau_day in zip(days, taus):
    print(day)
    df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    hd_data, S0 = HdData.filter_data(date=day)
    fig3 = plot_2d(df_tau, day, tau_day, hd_data, S0, x=x)
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
