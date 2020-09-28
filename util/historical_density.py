import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import integrate


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
    print(integrate.simps(y, x))
    print(np.trapz(y, x))