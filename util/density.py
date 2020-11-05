import numpy as np
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity
from util.smoothing import bspline


def density_estimation(sample, S, h, kernel="epanechnikov"):
    """
    Kernel Density Estimation for domain S, based on sample
    ------- :
    sample  : observed sample which density will be calculated
    S       : domain for which to calculate the sample for
    h       : bandwidth for KDE
    kernel  : kernel for KDE
    ------- :
    return  : density
    """
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(sample.reshape(-1, 1))
    log_dens = kde.score_samples(S.reshape(-1, 1))
    density = np.exp(log_dens)
    return density


def integrate(x, y):
    return simps(y, x)


def density_trafo_K2M(K, q_K, S, moneyness="K_S", analyze=False):
    """
    ------- :
    K       : K-domain of density
    q_K     : density in K-domain
    S       : spot price since M = S/K  # TODO: think about how it is in RND
    ------- :
    return  : density
    """
    if analyze:
        print("in K: ", integrate(K, q_K))
    pars, q_K, points = bspline(K, q_K, 30)

    num = len(K)
    M = np.linspace(0.5, 1.5, num)
    q_M = np.zeros(num)
    for i, m in enumerate(M):
        if moneyness == "S_K":
            q_M[i] = S / (m ** 2) * q_K(S / m)
        elif moneyness == "K_S":
            q_M[i] = S * q_K(S * m)
    if analyze:
        print("in M: ", integrate(M, q_M))
    return M, q_M


def pointwise_density_trafo_K2M(K, q_K, S_vals, M_vals, moneyness="K_S"):
    """
    ------- :
    K       : K-domain of density
    q_K     : density in K-domain
    S       : spot price since M = S/K  # TODO: think about how it is in RND
    ------- :
    return  : density
    """
    pars, q_K, points = bspline(K, q_K, 15)

    points = len(M_vals)
    q_pointsM = np.zeros(points)

    for i, m, s in zip(range(points), M_vals, S_vals):
        if moneyness == "S_K":
            q_pointsM[i] = s / (m ** 2) * q_K(s / m)
        elif moneyness == "K_S":
            q_pointsM[i] = s * q_K(s * m)
    return q_pointsM
