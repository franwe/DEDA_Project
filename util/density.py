import numpy as np
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity

def density_estimation(sample, S, h, kernel='epanechnikov'):
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
    print(simps(y, x))
    print(np.trapz(y, x))
