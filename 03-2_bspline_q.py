import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from util.smoothing import local_polynomial, bspline
from util.risk_neutral_density_bu import spd_appfinance

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


def plot_rnds(d, tau_day, x=0.1, num=140):
    print('exclude values outside of {} - {} Moneyness - {}/{}'.format(1-x, 1+x,
                                                                       sum(d.M > 1+x) + sum(d.M <= 1-x), d.shape[0]))
    df = d[(d.M <= 1+x) & (d.M > 1-x)]

    # ---------------------------------------------------------------- TRADING DAYS
    a = df.groupby(by=['tau_day', 'date']).count()
    a = a.reset_index()
    days = a[a.tau_day == tau_day].date
    print('Option was traded on {} days.'.format(len(days)))

    # --------------------------------------------------------------------- 2D PLOT
    color = cm.rainbow(np.linspace(0,1,len(days)))
    y_pos = 0.99
    x_pos = 0.99
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)

    res = dict()
    for day, c in zip(days, color):
        # S0 = d_usd[d_usd.Date == day]['Adj.Close'].tolist()[0]
        df_tau = df[(df.tau_day == tau_day) & (df.date == day)]
        df_tau = df_tau.reset_index()
        df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)

        h = df_tau.shape[0] ** (-1 / 9)
        tau = df_tau.tau.iloc[0]
        r = 0
        print('{}: {} options -- M_mean: {} -- M_std: {}'. format(day, df_tau.shape[0],
                                                                  np.mean(df_tau.M).round(3),
                                                                  np.std(df_tau.M).round(3)))

        spd = spd_appfinance
        smoothing_method = local_polynomial
        smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                         gridsize=num, kernel='epak')

        result = spd(M, S, K, smile, first, second, r, tau)

        # first many bsplines to have exactly same domain (M)
        #
        # M = K/S0
        pars, spline, points = bspline(M, result[::-1], sections=20, degree=2)
        M_adapted = np.linspace(1-x, 1+x, num=300)
        q = spline(M_adapted)

        # second, few bpslines for time series
        pars, spline, points = bspline(M_adapted, q, sections=5, degree=2)

        ax2.scatter(points['x'], points['y'], c='r')
        # ax2.scatter(M, result[::-1], s=2)
        ax2.scatter(M, spline(M), c=c, s=2)
        # ax2.plot(M_adapted, q, c=c)
        ax2.plot(M_adapted, spline(M_adapted), c=c)
        ax2.text(x_pos, y_pos, str(day),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes, c=c)
        res.update({day: {'t': pars['t'], 'c': pars['c']}})
        y_pos -= 0.05

    return fig2, res


def plot_rnds_3d(d, tau_day, x=0.1, num=140):
    print('exclude values outside of {} - {} Moneyness - {}/{}'.format(1-x, 1+x,
                                                                       sum(d.M > 1+x) + sum(d.M <= 1-x), d.shape[0]))
    d = d[(d.M <= 1+x) & (d.M > 1-x)]

    # ---------------------------------------------------------------- TRADING DAYS
    a = d.groupby(by=['tau_day', 'date']).count()
    a = a.reset_index()
    days = a[a.tau_day == tau_day].date
    print('Option was traded on {} days.'.format(len(days)))

    all_days = pd.date_range(start=d.date.min(), end=d.date.max())
    # --------------------------------------------------------------------- 3D PLOT
    color = cm.rainbow(np.linspace(0,1,len(days)))

    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.add_subplot(111, projection='3d')

    y_pos = -1
    i = 0
    # for day, c, i in zip(days, color, range(0, len(days))):
    for day in all_days:
        y_pos += 1
        df_tau = d[(d.tau_day == tau_day) & (d.date == str(day.date()))]
        # S0 = d_usd[d_usd.Date == str(day.date())]['Adj.Close'].tolist()[0]

        if df_tau.shape[0] == 0:
            pass
        else:
            c = color[i]
            i += 1

            df_tau = df_tau.reset_index()
            df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)

            h = df_tau.shape[0] ** (-1 / 9)
            tau = df_tau.tau.iloc[0]
            r = 0
            print('{}: {} options -- M_mean: {} -- M_std: {}'
                  .format(str(day.date()), df_tau.shape[0], np.mean(df_tau.M).round(3),
                          np.std(df_tau.M).round(3)))

            spd = spd_appfinance
            smoothing_method = local_polynomial
            smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                             gridsize=num, kernel='epak')
            result = spd(M, S, K, smile, first, second, r, tau)

            # first many bsplines to have exactly same domain (M)
            #
            # M = S0 / K # TODO: M = S0/K
            pars, spline, points = bspline(M, result[::-1], sections=20,
                                           degree=2)
            M_adapted = np.linspace(1 - x, 1 + x, num=300)
            q = spline(M_adapted)

            # second, few bpslines for time series
            pars, spline, points = bspline(M_adapted, q, sections=5, degree=2)

            y_points = [y_pos] * len(points['x'])
            # ax3.scatter(points['x'], y_points, points['y'], c='r', s=3)

            y_first = [y_pos] * len(M)
            ax3.plot(M, y_first, spline(M), c=c)

            y_adapted = [y_pos] * len(M_adapted)
            ax3.plot(M_adapted, y_adapted, spline(M_adapted), c=c, ls=':')

            ax3.plot(M, y_adapted, spline(M), c=c)


    plt.yticks(rotation=90)
    new_locs = [i for i in range(0, len(all_days)) if i%5 == 0]
    new_labels = [str(day.date()) for i, day in zip(range(0, len(all_days)), all_days) if i%5 == 0]
    ax3.set_yticks(new_locs)
    ax3.set_yticklabels(new_labels)

    ax3.set_xlabel('Moneyness')
    ax3.set_zlim(0)
    return fig3

# ------------------------------------------------------------------------ MAIN
# ---------------------------------------------------------- LOAD DATA ---- RND
d = pd.read_csv(data_path + 'trades_clean.csv')

x = 0.1
num = 300

fig2, res = plot_rnds(d, tau_day=2, x=x, num=num)
fig3 = plot_rnds_3d(d, tau_day=2, x=x, num=num)

# wang, okrhin uniform confidance bands






# TODO: cant to M, because cant compare to hd then... HD is in S... also need RND in S
# TODO: or S/S0?


