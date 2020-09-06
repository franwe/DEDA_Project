import scipy.optimize as opt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

from util.historical_density import density_estimation

cwd = os.getcwd() + os.sep



def _ARMA(w, a, b, obs):
    T = len(obs)
    sigma2 = np.zeros(T)

    for i in range(T):
        if i == 0:
            sigma2[i] = w/(1-(a+b))
        else:
            sigma2[i] = w + a*obs[i-1]**2 + b*sigma2[i-1]
    return sigma2


def _loglik(pars, e):
    w, a, b = pars
    sigma2 = _ARMA(w, a, b, e)

    loglik = -np.sum(-np.log(sigma2) - (e**2)/sigma2)  # minus because will minimize

    return loglik


def get_returns(data, target, dt=1, mode='log'):
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - dt - 1, target].reset_index()
    second = data.loc[dt:, target].reset_index()
    historical_returns = (second / first)[target]
    if mode=='log':
        return np.log(historical_returns)
    elif mode=='linear':
        return historical_returns


def fit_simga(returns):
    pars = (0.1, 0.05, 0.92)
    res = opt.minimize(_loglik, pars, args=(returns),
                bounds=((0.0001, None), (0.0001, None), (0.0001, None)),
                options={'disp':False})
    return res



def GARCH(w, a, b, sigma2_0, ret_0, T):
    sigma2 = np.zeros(T)
    residuals = np.zeros(T)

    z = np.random.randn(T)  # TODO: which distribution?
    for i in range(T):
        if i == 0:
            sigma2[i] = w + a*ret_0**2 + b*sigma2_0
            residuals[i] = z[i]*np.sqrt(sigma2[i])
        else:
            sigma2[i] = w + a * residuals[i - 1] ** 2 + b * sigma2[i - 1]
            residuals[i] = z[i]*np.sqrt(sigma2[i])
    return sigma2, residuals


def _Spath(residuals, mu, S0):
    S = np.zeros(len(residuals))
    returns = mu + residuals
    for i in range(0, len(S)):
        if i==0:
            S[i] = S0
        else:
            S[i] = S[i-1] * np.exp(returns[i])
    return S


def simulate(w, a, b, mu, sigma2_0, ret_0, T, S0, M=10000):
    ST = np.zeros(M)
    ET = np.zeros(M)
    st = np.zeros(M)
    for i in range(0,M):
        s, e = GARCH(w, a, b, sigma2_0=sigma2_0, ret_0=ret_0, T=T)
        S = _Spath(e, mu, S0=S0)
        ST[i] = S[-1]
        ET[i] = e[-1]
        st[i] = s[-1]

    return ST, (s, e, S)



def simulate_hd(data, day, tau_day, target, x=0.2):
    S0 = d_usd.loc[d_usd.Date == day, target].iloc[0]

    returns = get_returns(data, target, mode='log')

    res_mine = fit_simga(returns)
    print(res_mine.x)
    w, a, b = res_mine.x

    sigma2 = _ARMA(w, a, b, returns)

    # Simulation
    mu = np.mean(sigma2)
    sigma2_0 = sigma2[-1]
    ret_0 = returns.iloc[-1] + mu
    print(mu, sigma2_0, ret_0, S0)

    ST, tup = simulate(w, a, b, mu, sigma2_0, ret_0, T=tau_day, S0=S0, M=100000)

    S_domain = np.linspace((1-x)*S0, (1+x)*S0, num=100)
    hd = density_estimation(ST, S=S_domain, h=0.02*S0, kernel='epanechnikov')
    return hd, S_domain, S0


# ------------------------------------------------------------------- LOAD DATA
data_path = cwd + 'data' + os.sep
d_usd = pd.read_csv(data_path + 'BTCUSDT.csv')

data = d_usd
target = 'Adj.Close'
day = '2020-03-18'
tau_day = 2

hd, S_domain, S0 = simulate_hd(data, day, tau_day, target)
plt.axvline(S0, linewidth=2, color='r')
plt.plot(S_domain, hd)



# ------------------------------------------------------------------------- RND
#
# from util.smoothing import local_polynomial, bspline
# from util.risk_neutral_density_bu import spd_sfe, spd_appfinance, spd_rookley
#
# d = pd.read_csv(data_path + 'trades_clean.csv')
#
# def RND(data, day, tau_day, x=0.2):
#     print('exclude values outside of {} - {} Moneyness - {}/{}'.format(1-x, 1+x,
#           sum(data.M > 1+x) + sum(data.M <= 1-x), data.shape[0]))
#     df = data[(data.M <= 1+x) & (data.M > 1-x)]
#
#     # ---------------------------------------------------------------- TRADING DAYS
#     a = df.groupby(by=['tau_day', 'date']).count()
#     a = a.reset_index()
#     days = a[a.tau_day == tau_day].date
#     print('Option was traded on {} days.'.format(len(days)))
#
#     # --------------------------------------------------------------------- 2D PLOT
#
#     df_tau = df[(df.tau_day == tau_day) & (df.date == day)]
#     df_tau = df_tau.reset_index()
#     df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
#
#     h = df_tau.shape[0] ** (-1 / 9)
#     tau = df_tau.tau.iloc[0]
#     r = 0
#     print('{}: {} options -- M_mean: {} -- M_std: {}'. format(day, df_tau.shape[0],
#                                                           np.mean(df_tau.M).round(3),
#                                                           np.std(df_tau.M).round(3)))
#
#     spd = spd_appfinance
#     smoothing_method = local_polynomial
#     smile, first, second, M, S, K = smoothing_method(df_tau, tau, h, h_t=0.1,
#                                                  gridsize=200, kernel='epak')
#     result = spd(M, S, K, smile, first, second, r, tau)
#
#     # first many bsplines to have exactly same domain (M)
#     pars, spline, points = bspline(K, result[::-1], sections=20, degree=2)
#
#     return K, spline(K)
#
# S, q = RND(d, day, tau_day)
# plt.plot(S, q)
# TODO: different results than in 02-4_rnd_hd_bspline




















#-------- https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html

from arch import arch_model
import datetime as dt
import arch.data.sp500
from matplotlib import pyplot as plt
from datetime import datetime

st = dt.datetime(1988, 1, 1)
en = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()
market = data['Adj Close']
market = d_usd['Adj.Close']
dates = d_usd.Date.apply(lambda d: datetime.strptime(d, '%Y-%m-%d'))
market.index = dates
returns = get_returns(d_usd, 'Adj.Close', dt=1)
returns.index = market.index[1:]

returns = 100 * market.pct_change().dropna()
ax = returns.plot()
xlim = ax.set_xlim(returns.index.min(), returns.index.max())

am = arch_model(returns)
res = am.fit(update_freq=5)
print(res.summary())
fig = res.plot(annualize='D')

# forecast
forecasts = res.forecast(horizon=tau_day+1, method='bootstrap')
sims = forecasts.simulations

x = np.arange(1,tau_day+2)
lines = plt.plot(x, sims.residual_variances[-1, ::tau_day+1].T, color='#9cb2d6', alpha=0.5)
lines[0].set_label('Simulated path')
line = plt.plot(x, forecasts.variance.iloc[-1].values, color='#002868')
line[0].set_label('Expected variance')
plt.gca().set_xticks(x)
plt.gca().set_xlim(1,tau_day+1)
legend = plt.legend()

# ----- https://arch.readthedocs.io/en/latest/univariate/forecasting.html
from arch import arch_model
import datetime as dt

data = arch.data.sp500.load()
market = data['Adj Close']
# market = d_usd['Adj.Close']
returns = 100 * market.pct_change().dropna()


am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

# split
split_date = dt.datetime(2010,1,1)
res = am.fit(last_obs=split_date)

# Analytical forcast
forecasts = res.forecast(horizon=5, start=split_date)
forecasts.variance[split_date:].plot()

# Simulation Forecast
forecasts = res.forecast(horizon=5, start=split_date, method='bootstrap')
forecasts.variance[split_date:].plot()
print(forecasts.variance.tail())
