import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from localreg import *

from util.data import RndDataClass, HdDataClass
from util.smoothing import local_polynomial, bspline
from util.risk_neutral_density_bu import spd_appfinance
from util.garch import simulate_hd

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


def plot_rnds_3d(tau_day, x=0.5, mode='both'):
    RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
    HdData = HdDataClass(data_path + 'BTCUSDT.csv')
    # ------------------------------------------------------------ TRADING DAYS
    a = RndData.complete.groupby(by=['tau_day', 'date']).count()
    a = a.reset_index()
    days = a[a.tau_day == tau_day].date
    print('Option was traded on {} days.'.format(len(days)))

    all_days = pd.date_range(start=RndData.complete.date.min(),
                             end=RndData.complete.date.max())
    # ----------------------------------------------------------------- 3D PLOT
    color = cm.rainbow(np.linspace(0,1,len(days)))

    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.add_subplot(111, projection='3d')

    y_pos = -1
    i = 0

    for day_ts in all_days:
        day = str(day_ts.date())
        y_pos += 1
        df_tau = RndData.filter_data(date=day, tau_day=tau_day)

        if df_tau.shape[0] == 0:
            pass
        else:
            c = color[i]
            i += 1

            h = df_tau.shape[0] ** (-1 / 9)
            tau = df_tau.tau.iloc[0]
            r = 0
            if mode in ['both', 'rnd']:
                print('{}: {} options -- M_mean: {} -- M_std: {}'
                      .format(day, df_tau.shape[0], np.mean(df_tau.M).round(3),
                              np.std(df_tau.M).round(3)))
                # ------------------------------------------ ROOKLEY COMPONENTS
                spd = spd_appfinance
                smoothing_method = local_polynomial
                smile, first, second, M, S, K = smoothing_method(df_tau, tau, h,
                                                                 h_t=0.1,
                                                                 gridsize=140,
                                                                 kernel='epak')

                # ---------------------------- B-SPLINE on SMILE, FIRST, SECOND
                pars, spline, points = bspline(M, smile, sections=8, degree=3)
                # derivatives
                first_fct = spline.derivative(1)
                second_fct = spline.derivative(2)

                df_tau['q'] = df_tau.apply(lambda row: spd(row.M, row.S, row.K,
                                                           spline(row.M),
                                                           first_fct(row.M),
                                                           second_fct(row.M),
                                                           r, tau), axis=1)

                a = df_tau.sort_values('M')
                M_df = a.M.values
                q_df = a.q.values
                y_points = [y_pos] * len(M_df)

                y2 = localreg(M_df, q_df, degree=2, kernel=tricube, width=0.05)
                ax3.plot(M_df, y_points, y2, '-', c=c)

            if mode in ['both', 'hd']:
                ls = '-' if mode=='hd' else ':'
                hd_data, S0 = HdData.filter_data(date=day)
                S_hd = np.linspace((1-x) * S0, (1+x) * S0, num=100)
                hd, S_hd = simulate_hd(hd_data, S0, tau_day, S_domain=S_hd)
                y_hd = [y_pos] * len(S_hd)
                ax3.plot(S_hd / S0, y_hd, hd, ls, c=c)

    plt.yticks(rotation=90)
    new_locs = [i for i in range(0, len(all_days)) if i%5 == 0]
    new_labels = [str(day.date()) for i, day in zip(range(0, len(all_days)),
                                                    all_days) if i%5 == 0]
    ax3.set_yticks(new_locs)
    ax3.set_yticklabels(new_labels)

    ax3.set_xlabel('Moneyness')
    ax3.set_zlim(0)
    return fig3



def plot_taus(day, x=0.5, mode='rnd'):
    RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
    HdData = HdDataClass(data_path + 'BTCUSDT.csv')

    a = RndData.complete.groupby(by=['tau_day', 'date']).count()
    a = a.reset_index()
    taus = a[a.date == day].tau_day
    print('{} - maturities: {}'.format(day, list(taus)))

    color = cm.rainbow(np.linspace(0,1,len(taus)))

    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.add_subplot(111, projection='3d')

    y_pos = -1
    i = 0

    t = list(taus)
    t.remove(0)
    for tau_day in t:
        print(tau_day)
        y_pos += 1
        df_tau = RndData.filter_data(date=day, tau_day=tau_day)

        c = color[i]
        i += 1

        h = df_tau.shape[0] ** (-1 / 9)
        tau = df_tau.tau.iloc[0]
        r = 0
        if mode in ['both', 'rnd']:
            print('{}: {} options -- M_mean: {} -- M_std: {}'
                  .format(day, df_tau.shape[0], np.mean(df_tau.M).round(3),
                          np.std(df_tau.M).round(3)))
            # ------------------------------------------ ROOKLEY COMPONENTS
            spd = spd_appfinance
            smoothing_method = local_polynomial
            smile, first, second, M, S, K = smoothing_method(df_tau, tau, h,
                                                             h_t=0.1,
                                                             gridsize=140,
                                                             kernel='epak')

            # ---------------------------- B-SPLINE on SMILE, FIRST, SECOND
            pars, spline, points = bspline(M, smile, sections=8, degree=3)
            # derivatives
            first_fct = spline.derivative(1)
            second_fct = spline.derivative(2)

            df_tau['q'] = df_tau.apply(lambda row: spd(row.M, row.S, row.K,
                                                       spline(row.M),
                                                       first_fct(row.M),
                                                       second_fct(row.M),
                                                       r, tau), axis=1)

            a = df_tau.sort_values('M')
            M_df = a.M.values
            q_df = a.q.values
            y_points = [y_pos] * len(M_df)

            y2 = localreg(M_df, q_df, degree=2, kernel=tricube, width=0.05)
            ax3.plot(M_df, y_points, y2, '-', c=c)

        if mode in ['both', 'hd']:
            ls = '-' if mode=='hd' else ':'
            hd_data, S0 = HdData.filter_data(date=day)
            S_hd = np.linspace((1-x) * S0, (1+x) * S0, num=100)
            hd, S_hd = simulate_hd(hd_data, S0, tau_day, S_domain=S_hd)
            y_hd = [y_pos] * len(S_hd)
            ax3.plot(S_hd / S0, y_hd, hd, ls, c=c)

    new_labels = [0]+t
    ax3.set_yticklabels(new_labels)

    ax3.set_xlabel('Moneyness')
    ax3.set_ylabel('Maturity')
    ax3.set_zlim(0)
    return fig3

# ------------------------------------------------------------------------ MAIN
x = 0.5
tau_day = 14

fig2 = plot_rnds_3d(tau_day=tau_day, x=x, mode='both')

RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
RndData.analyse('2020-03-20')
plot_taus('2020-03-20', x=0.5, mode='rnd')