import numpy as np
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity
import pandas as pd
import os

from util.garch import get_returns, simulate_GARCH_moving

cwd = os.getcwd() + os.sep
# garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep


def sampling(data, target, tau_day, S0, M=10000):
    """
    Creates a Bootstrap sample of size M for price at maturity based on
    historical returns.
    ------- :
    data    : dataframe of time series, already cut to time-window of interest
              needs to have at least target column
    target  : name of column of prices, e.g. 'Adj. Close'
    tau_day : maturity in days (since time series is in days)
    S0      : current price
    M       : how many samples to draw
    ------- :
    return  : M prices at maturities, drawn from sample of historical returns
    """
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - tau_day - 1, target].reset_index()
    second = data.loc[tau_day:, target].reset_index()
    historical_returns = (first / second)[target]
    print('MC based on ', len(historical_returns), ' samples')
    sampled_returns = np.random.choice(historical_returns, M, replace=True)
    return S0 * sampled_returns


def density_estimation(sample, S, h, kernel='epanechnikov'):
    """
    Kernel Density Estimation for domain S, based on sample
    ------- :
    sample  : observed sample which density will be calculated
    S       : domain for which to calculate the sample for
    h       : bandwidth for KDE
    kernel  : kernel for KDE
    ------- :
    return  : density
    """
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(sample.reshape(-1, 1))
    log_dens = kde.score_samples(S.reshape(-1, 1))
    density = np.exp(log_dens)
    return density


def integrate(x, y):
    print(simps(y, x))
    print(np.trapz(y, x))


class HdCalculator:
    def __init__(self, data, tau_day, date, S0, path, h=0.1):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.S0 = S0
        self.garch_data = path
        self.h = h
        self.filename = 'T-{}_{}_S-single.csv'.format(self.tau_day, self.date)

        self.S = None
        self.M = None
        self.q_M = None
        self.q_S = None


    def get_hd(self):
        if os.path.exists(self.garch_data + self.filename):
            pass
        else:
            log_returns = get_returns(self.data) * 100
            self.filename = simulate_GARCH_moving(log_returns, self.S0, self.tau_day, self.date, filename=self.filename)

        S_sim = pd.read_csv(os.path.join(self.garch_data, self.filename))
        sample = np.array(S_sim['S'])
        self.S = np.linspace(0.5 * self.S0, 1.5 * self.S0, num=100)
        self.q_S = density_estimation(sample, self.S, h=self.h * self.S0)

        self.M = np.linspace(0.5, 1.5, num=100)
        self.q_M = density_estimation(sample/self.S0, self.M, h=self.h)
