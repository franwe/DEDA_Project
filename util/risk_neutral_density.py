import numpy as np
from scipy.stats import norm

from util.smoothing import local_polynomial, bspline
from util.density import density_trafo_K2M, pointwise_density_trafo_K2M

def spd_appfinance(M, S, K, o, o1, o2, r, tau):
    """ from Applied Quant. Finance - Chapter 8
    """
    st = np.sqrt(tau)
    rt = r*tau
    ert = np.exp(rt)

    d1 = (np.log(M) + (r + 1/2 * o**2)*tau)/(o*st)
    d2 = d1 - o*st

    del_d1_M = 1/(M*o*st)
    del_d2_M = del_d1_M
    del_d1_o = -(np.log(M) + rt)/(o**2 *st) + st/2
    del_d2_o = -(np.log(M) + rt)/(o**2 *st) - st/2


    d_d1_M = del_d1_M + del_d1_o * o1
    d_d2_M = del_d2_M + del_d2_o * o1

    dd_d1_M = (-(1/(M*o*st))*(1/M + o1/o)
               + o2*(st/2 - (np.log(M) + rt)/(o**2*st))
               + o1 * (2*o1 * (np.log(M)+rt)/(o**3*st) - 1/(M*o**2*st))
               )
    dd_d2_M = (-(1/(M*o*st))*(1/M + o1/o)
               - o2*(st/2 + (np.log(M) + rt)/(o**2*st))
               + o1 * (2*o1 * (np.log(M)+rt)/(o**3*st) - 1/(M*o**2*st))
               )

    d_c_M = (norm.pdf(d1) * d_d1_M
            - 1/ert * norm.pdf(d2)/M * d_d2_M
            + 1/ert * norm.cdf(d2)/(M**2))
    dd_c_M = (norm.pdf(d1) * (dd_d1_M - d1 * (d_d1_M)**2)
              - norm.pdf(d2)/(ert*M) * (dd_d2_M - 2/M * d_d2_M - d2 * (d_d2_M)**2)
              - 2*norm.cdf(d2)/(ert * M**3))

    dd_c_K = dd_c_M * (M/K)**2 + 2 * d_c_M * (M/K**2)
    q = ert * S * dd_c_K

    return q


class RndCalculator:
    def __init__(self, data, tau_day, date, h_densfit=None, h_iv=None):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.h_iv = self._h(h_iv)
        self.r = 0
        self.h_densfit = self._h(h_densfit)

        self.tau = self.data.tau.iloc[0]
        self.K = None
        self.M = None
        self.q_M = None
        self.q_K = None
        self.M_smile = None
        self.smile = None
        self.first = None
        self.second = None
        self.f = None  #  might delete later

    def _h(self, h):
        if h is None:
            return self.data.shape[0] ** (-1 / 9) # TODO: rethink this!
        else:
            return h

    # ------------------------------------------------------------------ SPD NORMAL
    def fit_smile(self):
        X = np.array(self.data.M)
        Y = np.array(self.data.iv)
        self.smile, self.first, self.second, self.M_smile, self.f = local_polynomial(X, Y, self.h_iv)

    def rookley(self):
        spd = spd_appfinance
        # ---------------------------------------- B-SPLINE on SMILE, FIRST, SECOND
        pars, spline, points = bspline(self.M_smile, self.smile, sections=8, degree=3)
        # derivatives
        first_fct = spline.derivative(1)
        second_fct = spline.derivative(2)

        self.data['q'] = self.data.apply(lambda row: spd(row.M, row.S, row.K,
                                                   spline(row.M), first_fct(row.M),
                                                   second_fct(row.M),
                                                   self.r, self.tau), axis=1)

        # step 1: Rookley results (points in K-domain) - fit density curve
        X = np.array(self.data.K)
        Q = np.array(self.data.q)
        self.q_K, first, second, self.K, f = local_polynomial(X, Q,
                                                              h=self.h_densfit*np.mean(X),
                                                              kernel='epak')

        # step 2: transform density POINTS from K- to M-domain
        self.data['q_M'] = pointwise_density_trafo_K2M(self.K, self.q_K, self.data.S, self.data.M)

        # step 3: density points in M-domain - fit density curve
        X = np.array(self.data.M)
        Q = np.array(self.data.q_M)
        self.q_M, first, second, self.M, f = local_polynomial(X, Q,
                                                              h=self.h_densfit*np.mean(X),
                                                              kernel='epak')