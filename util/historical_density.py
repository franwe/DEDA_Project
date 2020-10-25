import numpy as np
import pandas as pd
import os
from os.path import join

from util.density import density_estimation
from util.garch import GARCH


class HdCalculator(GARCH):
    def __init__(
        self,
        data,
        S0,
        path,
        tau_day,
        date,
        overwrite=True,
        target="price",
        window_length=365,
        n=400,
        h=0.1,
        M=5000,
    ):
        self.data = data
        self.target = target
        self.S0 = S0
        self.tau_day = tau_day
        self.date = date
        self.path = path
        self.overwrite = overwrite
        self.log_returns = self._get_log_returns()
        self.M = M
        GARCH.__init__(
            self,
            data=self.log_returns,
            window_length=window_length,
            data_name=self.date,
            n=n,
            h=h,
        )
        self.filename = "T-{}_{}_Ksim.csv".format(self.tau_day, self.date)

    def _get_log_returns(self):
        n = self.data.shape[0]
        data = self.data.reset_index()
        first = data.loc[: n - 2, self.target].reset_index()
        second = data.loc[1:, self.target].reset_index()
        historical_returns = (second / first)[self.target]
        return np.log(historical_returns) * 100

    def _calculate_path(self, all_summed_returns, all_tau_mu):
        S_T = self.S0 * np.exp(all_summed_returns / 100 + all_tau_mu / 100)
        return S_T

    def get_hd(self, variate):
        print(self.filename)
        # simulate M paths
        if os.path.exists(self.path + self.filename) and (self.overwrite == False):
            print("-------------- use existing Simulations")
            pass
        else:
            print("-------------- create new Simulations")
            all_summed_returns, all_tau_mu = self.simulate_paths(
                self.tau_day, self.M, variate
            )
            self.ST = self._calculate_path(all_summed_returns, all_tau_mu)
            pd.Series(self.ST).to_csv(join(self.path, self.filename), index=False)

        self.ST = pd.read_csv(join(self.path, self.filename))
        S_arr = np.array(self.ST)
        self.K = np.linspace(self.S0 * 0.2, self.S0 * 1.8, 100)
        self.q_K = density_estimation(S_arr, self.K, h=self.S0 * self.h)
        # self.M, self.q_M = density_trafo_K2M(self.K, self.q_K, self.S0, analyze=True)
        self.M = np.linspace(0.5, 1.5, 100)
        M_arr = np.array(self.S0 / self.ST)
        self.q_M = density_estimation(M_arr, self.M, h=self.h)