import scipy.optimize as opt
import numpy as np

from util.historical_density import density_estimation


def _ARMA(w, a, b, obs):
    T = len(obs)
    sigma2 = np.zeros(T)

    for i in range(T):
        if i == 0:
            sigma2[i] = w/(1-(a+b))
        else:
            sigma2[i] = w + a*obs[i-1]**2 + b*sigma2[i-1]
    return sigma2


def _loglik(pars, r):
    w, a, b = pars
    sigma2 = _ARMA(w, a, b, r)

    loglik = -np.sum(-np.log(sigma2) - (r**2)/(2*sigma2))  # minus because will minimize
    return loglik


def get_returns(data, target='Adj.Close', dt=1, mode='log'):
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - dt - 1, target].reset_index()
    second = data.loc[dt:, target].reset_index()
    historical_returns = (second / first)[target]
    if mode=='log':
        return np.log(historical_returns)
    elif mode=='linear':
        return historical_returns


def fit_sigma(returns):
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


def simulate_hd(data, S0, tau_day, S_domain, target='Adj.Close'):
    returns = get_returns(data, target, mode='log')

    res = fit_sigma(returns)
    w, a, b = res.x

    sigma2 = _ARMA(w, a, b, returns)

    # Simulation
    mu = np.mean(sigma2)
    sigma2_0 = sigma2[-1]
    ret_0 = returns.iloc[-1] + mu
    print('mu: {}, sigma2_0: {}, ret_0: {}, S0: {}'.format(mu, sigma2_0, ret_0, S0))

    ST, tup = simulate(w, a, b, mu, sigma2_0, ret_0, T=tau_day, S0=S0, M=100000)

    hd = density_estimation(ST, S=S_domain, h=0.02*S0, kernel='epanechnikov')
    return hd, S_domain
