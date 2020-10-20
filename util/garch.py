from matplotlib import pyplot as plt
import numpy as np
from arch import arch_model
import time

from util.density import density_estimation


class GARCH:
    def __init__(self, data, horizon, window_length=365, n=400, h=0.1, M=5000):
        self.data = data  # timeseries (here: log_returns)
        self.horizon = horizon
        self.h = h
        self.M = M
        self.window_length = window_length
        self.n = n
        self.burnin = horizon
        self.no_of_paths_to_save = 50

    def _GARCH_fit(self, data, q=1, p=1):
        model = arch_model(data, q=q, p=p)
        res = model.fit(disp="off")

        pars = (
            res.params["mu"],
            res.params["omega"],
            res.params["alpha[1]"],
            res.params["beta[1]"],
        )
        std_err = (
            res.std_err["mu"],
            res.std_err["omega"],
            res.std_err["alpha[1]"],
            res.std_err["beta[1]"],
        )
        return res, pars, std_err

    def create_Z(self):

        start = self.data.shape[0] - self.window_length - self.n
        end = self.data.shape[0] - self.window_length

        parameters = np.zeros((self.n - 1, 4))
        parameter_bounds = np.zeros((self.n - 1, 4))
        z_process = []
        e_process = []
        sigma2_process = []
        for i in range(0, self.n - 1):
            window = self.data[start + i : end + i]
            data = window - np.mean(window)

            res, parameters[i, :], parameter_bounds[i, :] = self._GARCH_fit(data)

            _, omega, alpha, beta = [
                res.params["mu"],
                res.params["omega"],
                res.params["alpha[1]"],
                res.params["beta[1]"],
            ]
            if i == 0:
                print(omega, alpha, beta)
                sigma2_tm1 = omega / (1 - alpha - beta)
            else:
                sigma2_tm1 = sigma2_process[-1]

            e_t = data.tolist()[-1]  # last observed log-return, mean adjust.
            e_tm1 = data.tolist()[-2]  # previous observed log-return
            sigma2_t = omega + alpha * e_tm1 ** 2 + beta * sigma2_tm1
            z_t = e_t / np.sqrt(sigma2_t)

            e_process.append(e_t)
            z_process.append(z_t)
            sigma2_process.append(sigma2_t)

        self.parameters = parameters
        self.parameter_bounts = parameter_bounds
        self.e_process = e_process
        self.z_process = z_process
        self.sigma2_process = sigma2_process

        # ------------------------------------------- kernel density estimation
        self.z_values = np.linspace(min(self.z_process), max(self.z_process), 500)
        h_dyn = self.h * (np.max(z_process) - np.min(z_process))
        self.z_dens = density_estimation(
            np.array(z_process), np.array(self.z_values), h=h_dyn
        ).tolist()
        return ()

    def _GARCH_simulate(self, pars):
        """stepwise GARCH simulation until burnin + horizon

        Args:
            pars (tuple): (mu, omega, alpha, beta)

        Returns:
            [type]: [description]
        """
        mu, omega, alpha, beta = pars

        sigma2 = [omega / (1 - alpha - beta)]
        e = [self.data.tolist()[-1] - mu]  # last observed log-return mean adj.
        weights = self.z_dens / (np.sum(self.z_dens))

        for _ in range(self.horizon + self.burnin):
            sigma2_tp1 = omega + alpha * e[-1] ** 2 + beta * sigma2[-1]
            z_tp1 = np.random.choice(self.z_values, 1, p=weights)[0]
            e_tp1 = z_tp1 * np.sqrt(sigma2_tp1)
            sigma2.append(sigma2_tp1)
            e.append(e_tp1)
        return sigma2[-self.horizon :], e[-self.horizon :]

    def simulate_paths(self):
        pars = np.mean(self.parameters, axis=0).tolist()
        # TODO: option to variate pars, to check for robustness
        print(pars)

        save_sigma = np.zeros((self.no_of_paths_to_save, self.horizon))
        save_e = np.zeros((self.no_of_paths_to_save, self.horizon))
        all_summed_returns = np.zeros(self.M)
        all_tau_mu = np.zeros(self.M)
        tick = time.time()
        for i in range(self.M):
            if (i + 1) % (self.M * 0.1) == 0:
                print(
                    "{}/{} - runtime: {} min".format(
                        i + 1, self.M, round((time.time() - tick) / 60)
                    )
                )
            sigma2, e = self._GARCH_simulate(pars)
            all_summed_returns[i] = np.sum(e)
            all_tau_mu[i] = self.horizon * pars[0]
            if i < self.no_of_paths_to_save:
                save_sigma[i, :] = sigma2
                save_e[i, :] = e

        self.save_sigma = save_sigma
        self.save_e = save_e
        self.all_summed_returns = all_summed_returns
        self.all_tau_mu = all_tau_mu
        return all_summed_returns, all_tau_mu

    def plot_params(self, pars, bounds, CI=False):
        fig_pars, axes = plt.subplots(4, 1, figsize=(8, 6))

        for i, name in zip(range(0, 4), ["mu", "omega", "alpha", "beta"]):
            axes[i].plot(pars[:, i], label="arch.arch_model", c="b")
            if CI:
                axes[i].plot(
                    range(0, len(pars)),
                    (pars[:, i] + 1.96 * bounds[:, i]),
                    ls=":",
                    c="b",
                )
            if CI:
                axes[i].plot(
                    range(0, len(pars)),
                    (pars[:, i] - 1.96 * bounds[:, i]),
                    ls=":",
                    c="b",
                )
            axes[i].set_ylabel(name)
        axes[0].legend()
        return fig_pars
