import os
import pickle
from localreg import *

from util.data import RndDataClass
from util.smoothing import local_polynomial, bspline
from util.risk_neutral_density import spd_appfinance

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

# ------------------------------------------------------------------------ MAIN

# ------------------------------------------------------------------- LOAD DATA
day = '2020-03-11'
num = 140
x=0.5
r=0
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)

# TODO: This section in RndDataClass.analyse
a = RndData.complete.groupby(by=['tau_day', 'date']).count()
a = a.reset_index()
taus = a[a.date == day].tau_day
t = list(taus)
t.remove(0)

res = dict()
for tau_day in t:
    print(tau_day)
    df_tau = RndData.filter_data(date=day, tau_day=tau_day)
    h = df_tau.shape[0] ** (-1 / 9)
    tau = df_tau.tau.iloc[0]

    # -------------------------------------------------------------- SPD NORMAL
    spd = spd_appfinance
    smoothing_method = local_polynomial
    smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
                                                     gridsize=140,
                                                     kernel='epak')

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

    y2 = localreg(M_df, q_df, degree=2, kernel=tricube, width=0.05)

    # ----------------------------------------------------------- STORE RESULTS
    res.update({tau_day : {'df': df_tau[['M', 'iv', 'S', 'K', 'q']],
                'M': M,
                'smile': smile,
                'first': first,
                'second': second,
                'K': K,
                'M_df': M_df,
                'q': q_df,
                'y2': y2,
                'S': S
                }})

with open(data_path + 'results_' + day + '.pkl', 'wb') as f:
    pickle.dump(res, f)
