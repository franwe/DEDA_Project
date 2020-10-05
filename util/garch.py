import os
import matplotlib.pyplot as plt
import numpy as np
import time
from arch import arch_model

from util.density import density_estimation

cwd = os.getcwd() + os.sep


class GARCH:
    def __init__(self, data, horizon, burnin,
                 window_length=365, h=0.1, M=5000):
        self.data = data  # timeseries (here: log_returns)
        self.horizon = horizon
        self.h = h
        self.M = M
        self.window_length = window_length
        self.burnin = burnin

        self.z_values = None
        self.z_dens = None
        self.all_returns = None
        self.returns_paths = None
        self.sigma2_paths = None


    def _GARCH_fit(self, data):
        model = arch_model(data, p=1, q=1)
        res2 = model.fit(disp='off')

        pars = [res2.params.mu, res2.params.omega, res2.params['alpha[1]'],
                res2.params[
                    'beta[1]']]
        bounds = [res2.std_err.mu, res2.std_err.omega, res2.std_err['alpha[1]'],
                  res2.std_err['beta[1]']]
        return res2, pars, bounds


    def create_Z(self):
        start_idx = 0
        end_idx = self.data.shape[0] - self.window_length

        sigma2 = [0]
        z_process = []
        pars = np.zeros([self.window_length, 4])  # preallocate
        bounds = np.zeros([self.window_length, 4])  # preallocate
        for i in range(self.window_length):
            window = self.data[start_idx+i : end_idx+i]
            mean_adjusted = window - window.mean()
            res, pars[i, :], bounds[i, :] = self._GARCH_fit(mean_adjusted)
            x_t = window.tolist()[-1]
            mu_t = res.params.mu * window.tolist()[-2]  # TODO: also try window.mean()

            e_tm1 = mean_adjusted.tolist()[-2]
            sigma2_t = res.params.omega + res.params['alpha[1]'] * e_tm1**2 \
                       + res.params['beta[1]'] * sigma2[-1]

            z_t = (x_t - mu_t)/np.sqrt(sigma2_t)

            sigma2.append(sigma2_t)
            z_process.append(z_t)

        self.z_values = np.linspace(np.min(z_process), np.max(z_process), 500).tolist()
        h_dyn = self.h * (np.max(z_process)-np.min(z_process))
        self.z_dens = density_estimation(np.array(z_process),
                                         np.array(self.z_values),
                                         h=h_dyn).tolist()

        return sigma2, z_process, pars, bounds


    def _simulate_GARCH(self):
        steps = self.burnin + self.horizon
        start_idx = self.data.shape[0] - self.window_length
        window = self.data[start_idx:].tolist()

        weights = self.z_dens/(np.sum(self.z_dens))

        window = window[1:]
        mean_adjusted = window - np.mean(window)
        res, pars, bounds = self._GARCH_fit(mean_adjusted)
        sigma2 = [res.params['omega']/(1-res.params['alpha[1]']-res.params['beta[1]'])]
        for i in range(steps):
            window = window[1:]
            mean_adjusted = window - np.mean(window)
            # res, pars[i, :], bounds[i, :] = GARCH_fit(mean_adjusted)
            x_t = window[-1]
            mu_tp1 = res.params.mu * x_t
            e_t = mean_adjusted.tolist()[-1]

            sigma2_tp1 = res.params['omega'] + res.params['alpha[1]'] * e_t**2 \
                         + res.params['beta[1]'] * sigma2[-1]
            z_tp1 = np.random.choice(self.z_values, 1, p=weights)[0]
            x_tp1 = z_tp1 * np.sqrt(sigma2_tp1) + mu_tp1

            sigma2.append(sigma2_tp1)
            window.append(x_tp1)
        return sigma2[-self.horizon:], window[-self.horizon:]


    def simulate_paths(self, save_paths=50):
        S = []
        tick = time.time()
        returns_paths = np.zeros((save_paths, self.horizon))
        sigma2_paths = np.zeros((save_paths, self.horizon))
        all_returns = np.zeros((self.M, self.horizon))
        for i in range(self.M):
            if (i+1) % (self.M * 0.1) == 0:
                print('{}/{} - runtime: {} min'.format(i+1, self.M, round(
                    (time.time() - tick) / 60)))
            sigma2, returns = self._simulate_GARCH()
            # ST = self._S_path(returns)
            all_returns[i, :] = returns

            if i < save_paths:
                returns_paths[i, :] = returns
                sigma2_paths[i, :] = sigma2

        self.S = S
        self.returns_paths = returns_paths
        self.sigma2_paths = sigma2_paths
        self.all_returns = all_returns


    def plot_params(self, pars, bounds, CI=False):
        fig_pars, axes = plt.subplots(4, 1, figsize=(8, 6))

        for i, name in zip(range(0, 4), ['mu', 'omega', 'alpha', 'beta']):
            axes[i].plot(pars[:, i], label='arch.arch_model', c='b')
            if CI: axes[i].plot(range(0, len(pars)), (pars[:, i] + 1.96*bounds[:, i]), ls=':', c='b')
            if CI: axes[i].plot(range(0, len(pars)), (pars[:, i] - 1.96*bounds[:, i]), ls=':', c='b')
            axes[i].set_ylabel(name)
        axes[0].legend()
        return fig_pars


from util.data import HdDataClass

HdData = HdDataClass(os.path.join(cwd, 'data', '00-raw', 'BTCUSDT.csv'))
day = '2020-04-03'
tau_day = 15
hd_data, S0 = HdData.filter_data(date=day)

def get_log_returns(data, target='Adj.Close'):
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - 2, target].reset_index()
    second = data.loc[1:, target].reset_index()
    historical_returns = (second / first)[target]
    return np.log(historical_returns) * 100

def S_path(S0, returns):
    returns = returns/100
    return S0 * np.exp(sum(returns.T))

garch.all_returns
S = S_path(S0, garch.all_returns)

from util.density import density_estimation

S_domain = np.linspace(S0*0.5, S0*1.5, 200)
q = density_estimation(S, S_domain, h=S0*0.2)

plt.plot(S_domain, q)

log_returns = get_log_returns(hd_data)

garch = GARCH(log_returns, tau_day, burnin=20, M=100, h=0.2)
sigma2, z_process, pars, bounds = garch.create_Z()
garch.simulate_paths()

# plt.plot(garch.z_values, garch.z_dens)
plt.plot(garch.S_domain/S0, garch.q)

from statsmodels.graphics.gofplots import qqplot

qqplot(np.array(z_process), line='45')