import numpy as np
from scipy.stats import norm


def spd_rookley(M, S, X, o, o1, o2, r, tau):
    """
    Tried to implement from rookley paper
    :param M: moneyness
    :param S: price of underlying
    :param X: strike
    :param o: sigma
    :param o1: first derivative of sigma
    :param o2: second derivative of sigma
    :return: risk neutral density
    """

    ert = np.exp(-r*tau)
    st = np.sqrt(tau)
    rt = r*tau

    d1 = (np.log(M) + (r + 1/2 * o**2)*tau)/(o*st)
    d2 = d1 - o*st

    # partial derivatives of d1 and d2
    d1_M = 1/(M*o*st)
    d2_M = d1_M

    d1_o = -(np.log(M)+rt)/(o**2*st) + st/2
    d2_o = -(np.log(M)+rt)/(o**2*st) - st/2

    # total first derivatives of d1 and d2
    d_d1_M = d1_M + d1_o * o1  # curved d, partial derivatives
    d_d2_M = d2_M + d2_o * o1

    # total second derivates of d1 and d2

    dd_d1_M = (-1/(M*o*st) * (1/M + o1/o)
               + o2 * (st/2 - (np.log(M) + rt)/(o**2-st))
               + o1 * (2*o1*(np.log(M)+rt)/(o**3*st) - 1/(M*(o**2)*st))
               )

    dd_d2_M = (-1/(M*o*st) * (1/M + o1/o)
               + o2 * (st/2 + (np.log(M) + rt)/(o**2-st))
               + o1 * (2*o1*(np.log(M)+rt)/(o**3*st) - 1/(M*(o**2)*st))
               )

    # total derivatives of c
    d_c_M = (norm.pdf(d1) * d_d1_M - (ert * norm.pdf(d2))/M * d_d2_M
             + (ert * norm.cdf(d2))/(M**2)
             )
    dd_c_M = (norm.pdf(d1) * (dd_d1_M - d1*(d_d1_M**2))
              - ((ert * norm.pdf(d2))/M) * (dd_d2_M -
                                          (2/M) * d_d2_M - d2*(d_d2_M**2))
              - ((2*ert*norm.cdf(d2))/(M**3))
              )
    dd_c_X = dd_c_M * (M/X)**2 + 2 * d_c_M * (M/X**2)

    q = np.exp(r*tau) * S * dd_c_X
    return q


def spd_appfinance(M, S, K, o, o1, o2, r, tau):
    """ best bet - from Applied Quant. Finance - Chapter 8
        almost same as spdbl but smaller scaled?
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

    # d_c_S = d_c_M * 1/K
    # dd_c_S = dd_c_M * (1/K)**2

    # dd_C_s = 2* d_c_S + S*dd_c_S

    dd_c_K = dd_c_M * (M/K)**2 + 2 * d_c_M * (M/K**2)
    q = ert * S * dd_c_K
    return q


def spd_sfe(m, s, X, sigma, sigma1, sigma2, r, tau):
    """ from SFE_RND_HD """
    rm = len(m)
    st = np.sqrt(tau)
    ert = np.exp(r * tau)
    rt = r * tau

    # Modified Black-Scholes scaled by S-div instead of F
    d1 = (np.log(m) + tau * (r + 0.5 * (sigma ** 2))) / (sigma * st)
    d2 = d1 - sigma * st

    f = norm.cdf(d1) - norm.cdf(d2) / (ert * m)

    # First derivative of d1 term
    d11 = (1 / (m * sigma * st)) \
          - (1 / (st * (sigma ** 2))) * ((np.log(m) + tau * r) * sigma1) \
          + 0.5 * st * sigma1

    # First derivative of d2 term
    d21 = d11 - st * sigma1

    # Second derivative of d1 term
    d12 = -(1 / (st * (m ** 2) * sigma)) \
          - sigma1 / (st * m * (sigma ** 2)) + sigma2 \
          * (0.5 * st - (np.log(m) + rt) / (st * (sigma ** 2))) \
          + sigma1 * (2 * sigma1 *(np.log(m) + rt) / (st * sigma ** 3)
                      - 1 / (st * m * sigma ** 2))

    # Second derivative of d2 term
    d22 = d12 - st * sigma2

    # Please refer to either Rookley (1997) or the XploRe Finance Guide for
    # derivations
    f1 = norm.pdf(d1) * d11 + (1 / ert) \
         * ((-norm.pdf(d2) * d21) / m
            + norm.cdf(d2) / (m ** 2))
    f2 = norm.pdf(d1) * d12 - d1 * norm.pdf(d1) \
         * (d11 ** 2) - (1 / (ert * m) * norm.pdf(d2) * d22) \
         + ((norm.pdf(d2) * d21) / ( ert * m ** 2)) \
         + (1 / (ert * m) * d2 * norm.pdf(d2) * (d21 ** 2)) \
         - (2 * norm.cdf(d2) / (ert * (m ** 3))) \
         + (1 / (ert * (m ** 2)) * norm.pdf(d2) * d21)

    # Recover strike price
    x = s / m
    c1 = -(m ** 2) * f1
    c2 = s * ((1 / x ** 2) * ((m ** 2) * f2 + 2 * m * f1))

    # Calculate the quantities of interest
    cdf = ert * c1 + 1
    fstar = (ert * c2)

    return fstar
