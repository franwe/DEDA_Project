import numpy as np
from scipy.stats import norm


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
