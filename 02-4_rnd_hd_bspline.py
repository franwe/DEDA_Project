import os
from matplotlib import pyplot as plt

from util.data import RndDataClass, HdDataClass
from util.smoothing import local_polynomial, bspline
from util.risk_neutral_density_bu import spd_appfinance
from util.expand import expand_X
from util.garch import simulate_hd

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


# --------------------------------------------------------------------- 2D PLOT
def plot_2d(df_tau, day, tau_day, hd_data, S0):
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]
    r = 0

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    # ------------------------------------------------------------------ SPD NORMAL
    spd = spd_appfinance
    smoothing_method = local_polynomial
    smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                 gridsize=140, kernel='epak')
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
    hd, S_hd = simulate_hd(hd_data, S0, tau_day)
    ax3.plot(S_hd, hd, '-', c='b')

    ax3.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax3.transAxes)

    return fig3, df_tau


# -----------------------------------------------------------------------------
day = '2020-03-06'
tau_day = 21
x = 0.3

# ----------------------------------------------------------- LOAD DATA HD, RND
HdData = HdDataClass(data_path + 'BTCUSDT.csv')
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
# TODO: Cutoff has huge influence!!!!

# ----------------------------------------------------------------------- Plots
days = ['2020-03-11', '2020-03-20', '2020-03-29', '2020-03-06']
taus = [2,             2,            2,            21]

for day, tau_day in zip(days, taus):
    df_new = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
    hd_data, S0 = HdData.filter_data(date=day)
    fig3, new = plot_2d(df_new, day, tau_day, hd_data, S0)

day = '2020-03-29'
tau_day = 2
df_new = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
hd_data, S0 = HdData.filter_data(date=day)
fig3 = plot_2d(df_new, day, tau_day, hd_data)