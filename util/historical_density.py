import numpy as np
import pandas as pd
import os
from os.path import join

from util.density import density_estimation, density_trafo_K2M
from util.garch import GARCH


class HdCalculator:
    def __init__(self, data, tau_day, date, S0, burnin, path, M=5000, target='Adj.Close', h=0.2, overwrite=False):
        self.data = data
        self.target = target
        self.tau_day = tau_day
        self.date = date
        self.S0 = S0
        self.garch_data = path
        self.h = h
        self.burnin = burnin
        self.M_simulations = M
        self.filename = 'T-{}_{}_Ksim.csv'.format(self.tau_day, self.date)
        self.overwrite = overwrite

        self.log_returns = None   # log-returns of spot prices
        self.S = None             # simulated S_T
        self.K = None             # K-domain for density
        self.M = None             # M-domain for density
        self.q_K = None           # density in K-domain
        self.q_M = None           # density in M-domain

        self._get_log_returns()
        self.GARCH = GARCH(self.log_returns, self.tau_day, burnin=self.burnin,
                           M=self.M_simulations, h=self.h)

    def _get_log_returns(self):
        n = self.data.shape[0]
        data = self.data.reset_index()
        first = data.loc[:n - 2, self.target].reset_index()
        second = data.loc[1:, self.target].reset_index()
        historical_returns = (second / first)[self.target]
        self.log_returns = np.log(historical_returns) * 100

    def _S_paths(self, S0, log_returns):
        log_returns = log_returns / 100
        self.S = S0 * np.exp(sum(log_returns.T))

    def get_hd(self):
        print(self.filename)
        # simulate M paths
        if os.path.exists(self.garch_data + self.filename) and (self.overwrite == False):
            print('use existing file')
            pass
        else:
            print('create new file')
            sigma2, z_process = self.GARCH.create_Z()
            self.GARCH.simulate_paths()
            self._S_paths(self.S0, self.GARCH.all_returns)
            pd.Series(self.S).to_csv(join(self.garch_data, self.filename), index=False)

        self.S = pd.read_csv(join(self.garch_data, self.filename))
        S_arr = np.array(self.S)
        self.K = np.linspace(self.S0 * 0.2, self.S0 * 1.8, 500)
        self.q_K = density_estimation(S_arr, self.K,  h=self.S0 * self.h)
        # self.M, self.q_M = density_trafo_K2M(self.K, self.q_K, self.S0, analyze=True)
        self.M = np.linspace(0.5, 1.5, 500)
        M_arr = np.array(self.S0 / self.S)
        self.q_M = density_estimation(M_arr, self.M, h=self.h)
