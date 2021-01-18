import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from util.smoothing import (
    create_fit,
    bandwidth_cv,
    bspline,
    local_polynomial_estimation,
    linear_estimation,
)
from util.density import pointwise_density_trafo_K2M
from statsmodels.nonparametric.bandwidths import bw_silverman
import os
from os.path import join
import pandas as pd

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-1_rnd") + os.sep


def create_bandwidth_range(X, bins_max=30, num=10):
    bw_silver = bw_silverman(X)
    if bw_silver > 10:
        lower_bound = max(0.5 * bw_silver, 100)
    else:
        lower_bound = max(0.5 * bw_silver, 0.03)
    x_bandwidth = np.linspace(lower_bound, 7 * bw_silver, num)
    print("------ Silverman: ", bw_silver)
    return x_bandwidth, bw_silver, lower_bound


def spd_appfinance(M, S, K, o, o1, o2, r, tau):
    """from Applied Quant. Finance - Chapter 8"""
    st = np.sqrt(tau)
    rt = r * tau
    ert = np.exp(rt)

    d1 = (np.log(M) + (r + 1 / 2 * o ** 2) * tau) / (o * st)
    d2 = d1 - o * st

    del_d1_M = 1 / (M * o * st)
    del_d2_M = del_d1_M
    del_d1_o = -(np.log(M) + rt) / (o ** 2 * st) + st / 2
    del_d2_o = -(np.log(M) + rt) / (o ** 2 * st) - st / 2

    d_d1_M = del_d1_M + del_d1_o * o1
    d_d2_M = del_d2_M + del_d2_o * o1

    dd_d1_M = (
        -(1 / (M * o * st)) * (1 / M + o1 / o)
        + o2 * (st / 2 - (np.log(M) + rt) / (o ** 2 * st))
        + o1
        * (2 * o1 * (np.log(M) + rt) / (o ** 3 * st) - 1 / (M * o ** 2 * st))
    )
    dd_d2_M = (
        -(1 / (M * o * st)) * (1 / M + o1 / o)
        - o2 * (st / 2 + (np.log(M) + rt) / (o ** 2 * st))
        + o1
        * (2 * o1 * (np.log(M) + rt) / (o ** 3 * st) - 1 / (M * o ** 2 * st))
    )

    d_c_M = (
        norm.pdf(d1) * d_d1_M
        - 1 / ert * norm.pdf(d2) / M * d_d2_M
        + 1 / ert * norm.cdf(d2) / (M ** 2)
    )
    dd_c_M = (
        norm.pdf(d1) * (dd_d1_M - d1 * (d_d1_M) ** 2)
        - norm.pdf(d2)
        / (ert * M)
        * (dd_d2_M - 2 / M * d_d2_M - d2 * (d_d2_M) ** 2)
        - 2 * norm.cdf(d2) / (ert * M ** 3)
    )

    dd_c_K = dd_c_M * (M / K) ** 2 + 2 * d_c_M * (M / K ** 2)
    q = ert * S * dd_c_K

    return q


class RndCalculator:
    def __init__(self, data, tau_day, date, h_m=None, h_m2=None, h_k=None):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.h_m = h_m
        self.h_m2 = h_m2
        self.h_k = h_k
        self.r = 0

        self.tau = self.data.tau.iloc[0]
        self.K = None
        self.M = None
        self.q_M = None
        self.q_K = None
        self.M_smile = None
        self.smile = None
        self.first = None
        self.second = None

    # -------------------------------------------------------------- SPD NORMAL
    def bandwidth_and_fit(self, X, y):
        x_bandwidth, bw_silver, lower_bound = create_bandwidth_range(X)
        cv_results = bandwidth_cv(
            X, y, x_bandwidth, smoothing=local_polynomial_estimation
        )[0]
        cv = cv_results[1]
        h = cv_results[2]

        fit, first, second, X_domain = create_fit(X, y, h)

        return (h, x_bandwidth, cv), (fit, first, second, X_domain)

    def fit_smile(self):
        X = np.array(self.data.M)
        y = np.array(self.data.iv)
        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        self.h_m = res_bandwidth[0]
        self.smile, self.first, self.second, self.M_smile = res_fit
        return

    def fit_smile_corrected(self):
        X = np.array(self.data.M)
        y = np.array(self.data.iv)
        self.smile, self.first, self.second, self.M_smile = create_fit(
            X, y, self.h_m
        )
        return

    def rookley(self):
        spd = spd_appfinance
        # ------------------------------------ B-SPLINE on SMILE, FIRST, SECOND
        print("fit bspline to derivatives for rookley method")
        pars, spline, points = bspline(
            self.M_smile, self.smile, sections=8, degree=3
        )
        # derivatives
        first_fct = spline.derivative(1)
        second_fct = spline.derivative(2)

        # step 1: calculate spd for every option-point "Rookley's method"
        print("calculate q_K (Rookley Method)")
        self.data["q"] = self.data.apply(
            lambda row: spd(
                row.M,
                row.S,
                row.K,
                spline(row.M),
                first_fct(row.M),
                second_fct(row.M),
                self.r,
                self.tau,
            ),
            axis=1,
        )

        # step 2: Rookley results (points in K-domain) - fit density curve
        print("locpoly fit to rookley result q_K")
        X = np.array(self.data.K)
        y = np.array(self.data.q)

        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        self.h_k = res_bandwidth[0]
        (
            self.q_K,
            _first,
            _second,
            self.K,
        ) = res_fit

        # step 3: transform density POINTS from K- to M-domain
        print("density transform rookley points q_K to q_M")
        self.data["q_M"] = pointwise_density_trafo_K2M(
            self.K, self.q_K, self.data.S, self.data.M
        )

        # step 4: density points in M-domain - fit density curve
        print("locpoly fit to q_M")
        X = np.array(self.data.M)
        y = np.array(self.data.q_M)
        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        self.h_m2 = res_bandwidth[0]
        (
            self.q_M,
            _first,
            _second,
            self.M,
        ) = res_fit

        bandwidths = pd.DataFrame(
            [self.date, self.tau_day, self.h_m, self.h_m2, self.h_k]
        ).T
        bandwidths.to_csv(
            join(save_data, "bandwidths.csv"),
            index=False,
            header=False,
            mode="a",
        )
        return

    def rookley_corrected(self):
        spd = spd_appfinance
        # ------------------------------------ B-SPLINE on SMILE, FIRST, SECOND
        pars, spline, points = bspline(
            self.M_smile, self.smile, sections=8, degree=3
        )
        # derivatives
        first_fct = spline.derivative(1)
        second_fct = spline.derivative(2)

        # step 1: calculate spd for every option-point "Rookley's method"
        self.data["q"] = self.data.apply(
            lambda row: spd(
                row.M,
                row.S,
                row.K,
                spline(row.M),
                first_fct(row.M),
                second_fct(row.M),
                self.r,
                self.tau,
            ),
            axis=1,
        )

        # step 2: Rookley results (points in K-domain) - fit density curve
        X = np.array(self.data.K)
        y = np.array(self.data.q)

        (
            self.q_K,
            _first,
            _second,
            self.K,
        ) = create_fit(X, y, self.h_k)

        # step 3: transform density POINTS from K- to M-domain
        self.data["q_M"] = pointwise_density_trafo_K2M(
            self.K, self.q_K, self.data.S, self.data.M
        )

        # step 4: density points in M-domain - fit density curve
        X = np.array(self.data.M)
        y = np.array(self.data.q_M)
        (
            self.q_M,
            _first,
            _second,
            self.M,
        ) = create_fit(X, y, self.h_m2)
        return

    def calc_deriv(self, option):
        df = self.data[self.data.option == option]
        X = np.array(df.K)
        y = np.array(df.P)

        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        h = res_bandwidth[0]

        fit, first, second, K_domain = create_fit(X, y, h)

        return K_domain, second

    def d2C_dK2(self):
        C_domain, C2 = self.calc_deriv(option="C")
        P_domain, P2 = self.calc_deriv(
            option="P"
        )  # TOdO: puts give wave-result. somehow need smoother fit of prices. or S0=True?
        X = np.append(C_domain, P_domain)
        y = np.append(C2, P2)
        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        fit, first, second, K_domain = res_fit

        from matplotlib import pyplot as plt

        plt.plot(C_domain, C2, ls=":")
        plt.plot(P_domain, P2, ls=":")
        plt.plot(K_domain, fit, "k")
        plt.show()
        S0 = self.data.S[0]
        self.M_num = S0 / K_domain

        self.q_M_num = pointwise_density_trafo_K2M(
            K_domain, fit, np.ones(100) * S0, self.M_num
        )


def plot_rookleyMethod(RND, x=0.5):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(12, 4))

    # smile
    ax0.scatter(RND.data.M, RND.data.iv, c="r", s=4)
    ax0.plot(RND.M_smile, RND.smile)
    ax0.set_xlabel("Moneyness")
    ax0.set_ylabel("implied volatility")
    ax0.set_xlim(1 - x, 1 + x)

    # derivatives
    ax1.plot(RND.M_smile, RND.smile)
    ax1.plot(RND.M_smile, RND.first)
    ax1.plot(RND.M_smile, RND.second)
    ax1.set_xlabel("Moneyness")
    ax1.set_xlim(1 - x, 1 + x)

    # density q_k
    ax2.scatter(RND.data.K, RND.data.q, c="r", s=4)
    ax2.plot(RND.K, RND.q_K)
    ax2.set_xlabel("Strike Price")
    ax2.set_ylabel("risk neutral density")

    # density q_m
    ax3.scatter(RND.data.M, RND.data.q_M, c="r", s=4)
    ax3.plot(RND.M, RND.q_M)
    ax3.set_xlabel("Moneyness")
    ax3.set_ylabel("risk neutral density")
    ax3.set_xlim(1 - x, 1 + x)
    return fig
