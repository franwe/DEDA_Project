import numpy as np
from scipy.stats import norm

from util.smoothing import local_polynomial, bspline

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
    def __init__(self, data, tau_day, date, h=None):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.h = h
        self._h()  # parse h
        self.r = 0

        self.tau = self.data.tau.iloc[0]
        self.K = None
        self.M_smile = None
        self.q_fitM = None
        self.q_fitK = None
        self.smile = None
        self.first = None
        self.second = None
        self.f = None  #  might delete later

    def _h(self):
        if self.h is None:
            print('if')
            self.h = self.data.shape[0] ** (-1 / 9) # TODO: rethink this!
        else:
            print('else')

    # ------------------------------------------------------------------ SPD NORMAL
    def fit_smile(self):
        X = np.array(self.data.M)
        Y = np.array(self.data.iv)
        self.smile, self.first, self.second, self.M_smile, self.f = local_polynomial(X, Y, self.h)

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
        X = np.array(self.data.M)
        Q = np.array(self.data.q)
        self.q_fitM, first, second, self.M, f = local_polynomial(X, Q, h=0.1,
                                                           kernel='epak')

        X = np.array(self.data.K)
        Q = np.array(self.data.q)
        self.q_fitK, first, second, self.K, f = local_polynomial(X, Q, h=0.1*np.mean(X),
                                                           kernel='epak')