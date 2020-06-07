import numpy as np
from sklearn.neighbors import KernelDensity


def SVCJ(tau_day, S0, M=10000, myseed=12345):
    # SVCJ parameters
    mu = 0.042
    r = mu
    mu_y = -0.0492
    sigma_y = 2.061
    l = 0.0515
    alpha = 0.0102
    beta = -0.188
    rho = 0.275
    sigma_v = 0.007
    rho_j = -0.210
    mu_v = 0.709
    v0 = 0.19 ** 2
    kappa = 1 - beta
    theta = alpha / kappa

    np.random.seed(myseed)
    dt = 1 / tau_day  # dt
    m = int(tau_day * (1 / dt) / tau_day)  # time horizon in days

    # trialrun
    # dt     = 1/10
    # n      = 1000
    # m      = int(10*(1/dt)/10)

    T = m * dt
    t = np.arange(0, T + dt, dt)

    w = np.random.standard_normal([M, m])
    w2 = rho * w + np.sqrt(1 - rho ** 2) * np.random.standard_normal([M, m])
    z_v = np.random.exponential(mu_v, [M, m])
    z_y = np.random.standard_normal([M, m]) * sigma_y + mu_y + rho_j * z_v
    dj = np.random.binomial(1, l * dt, size=[M, m])
    s = np.zeros([M, m + 1])
    v = np.zeros([M, m + 1])

    s0 = S0
    s[:, 0] = s0  # initial CRIX level, p. 20
    v[:, 0] = v0

    for i in range(1, m + 1):
        v[:, i] = v[:, i - 1] \
                  + kappa * (theta - np.maximum(0, v[:, i - 1])) \
                  * dt \
                  + sigma_v * np.sqrt(np.maximum(0, v[:, i - 1])) \
                  * w2[:, i - 1] \
                  + z_v[:, i - 1] * dj[:, i - 1]
        s[:, i] = s[:, i - 1] * (1 + (r - l * (mu_y + rho_j * mu_v)) * dt
                                 + np.sqrt(v[:, i - 1] * dt) * w[:, i - 1]) \
                  + z_v[:, i - 1] * dj[:, i - 1]
    sample = s[:,-1]
    return sample, (s, v)


def MC_return(data, target, tau_day, S0, M=10000):
    n = data.shape[0]
    first = data.loc[:n - tau_day - 1, target].reset_index()
    second = data.loc[tau_day:, target].reset_index()
    historical_returns = S0 * (first / second)[target]
    print('MC based on ', len(historical_returns), ' samples')
    sample = np.random.choice(historical_returns, M, replace=True)
    return sample


def density_estimation(sample, S, S0, h, kernel='epanechnikov'):
    h = h * S0
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(sample.reshape(-1, 1))
    log_dens = kde.score_samples(S.reshape(-1, 1))
    hd = np.exp(log_dens)
    return hd