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


def create_bandwidth_range(X, bins_max=60):
    X_range = X.max() - X.min()
    x_bandwidth = np.linspace(X_range / bins_max, X_range / 5, num=50)
    return x_bandwidth


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
    def __init__(self, data, tau_day, date, h_densfit=None, h_iv=None):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.h_m = None
        self.h_k = None
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
        x_bandwidth = create_bandwidth_range(X)
        cv = bandwidth_cv(
            X, y, x_bandwidth, smoothing=local_polynomial_estimation
        )
        h = x_bandwidth[cv.argmin()]
        fit, first, second, X_domain = create_fit(X, y, h)
        return (h, x_bandwidth, cv), (fit, first, second, X_domain)

    def rookley(self):
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

        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        self.h_k = res_bandwidth[0]
        (
            self.q_K,
            _first,
            _second,
            self.K,
        ) = res_fit

        # step 3: transform density POINTS from K- to M-domain
        self.data["q_M"] = pointwise_density_trafo_K2M(
            self.K, self.q_K, self.data.S, self.data.M
        )

        # step 4: density points in M-domain - fit density curve
        X = np.array(self.data.M)
        y = np.array(self.data.q_M)
        res_bandwidth, res_fit = self.bandwidth_and_fit(X, y)
        self.h_m2 = res_bandwidth[0] * 1.5
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


def plot_rookleyMethod(RND):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))

    # smile
    ax0.scatter(RND.data.M, RND.data.iv, c="r", s=4)
    ax0.plot(RND.M_smile, RND.smile)
    ax0.set_xlabel("Moneyness")
    ax0.set_ylabel("implied volatility")

    # derivatives
    ax1.plot(RND.M_smile, RND.smile)
    ax1.plot(RND.M_smile, RND.first)
    ax1.plot(RND.M_smile, RND.second)
    ax1.set_xlabel("Moneyness")

    # density
    ax2.scatter(RND.data.M, RND.data.q_M, c="r", s=4)
    ax2.plot(RND.M, RND.q_M)
    ax2.set_xlabel("Moneyness")
    ax2.set_ylabel("risk neutral density")
    return fig
