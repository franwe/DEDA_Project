import numpy as np
from scipy.stats import norm

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


# --------------------------------------------------------- Quantlet SFE_RND_HD
def locpoly_r(df, h, h_t=0.1, gridsize=400, kernel=None):
    """
    smoothing using the R (Fortran) locpoly function from KernSmooth package
    """

    KernSmooth = importr('KernSmooth')
    rloc = ro.r['locpoly']

    m = ro.FloatVector(list(df.M))  # Todo: need to standardize?
    m = ro.FloatVector(list(df.M_std))
    rx = ro.FloatVector([min(df.M_std), max(df.M_std)])
    iv = ro.FloatVector(list(df.iv))

    out = rloc(x=m, y=iv, bandwidth=h, degree=2,
               gridsize=gridsize, range_x=rx)
    x = np.array(list(out[0]))
    smile = np.array(list(out[1]))

    out = rloc(x=m, y=iv, bandwidth=h, degree=2, drv=1,
               gridsize=gridsize, range_x=rx)
    first = np.array(list(out[1]))/np.std(df.M)

    out = rloc(x=m, y=iv, bandwidth=h, degree=2, drv=2,
               gridsize=gridsize, range_x=rx)
    second = np.array(list(out[1]))/np.std(df.M)
    M_min, M_max = min(df.M), max(df.M)
    M_std_min, M_std_max = min(df.M_std), max(df.M_std)
    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)

    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)
    M = np.linspace(M_min, M_max, gridsize)
    M_std = np.linspace(M_std_min, M_std_max, gridsize)
    return smile, first, second, M, S, K, M_std


# ---------------------------------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return norm.cdf(u_m) * norm.cdf(u_t)


def epanechnikov(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return 3/4 * (1-u_m)**2 * 3/4 * (1-u_t)**2


def smoothing_rookley(df, m, t, h_m, h_t, kernel=gaussian_kernel):
    # M = np.array(df.M)
    M = np.array(df.M_std)
    T = np.array(df.tau)
    y = np.array(df.iv)
    n = df.shape[0]

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X4 = T-t
    X5 = (T-t)**2
    X6 = X2*X4
    X = np.array([X1, X2, X3, X4, X5, X6]).T

    ker = kernel(M, m, h_m, T, t, h_t)
    W = np.diag(ker)

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    return beta[0], beta[1], 2*beta[2]


def rookley(df, h_m, h_t=0.1, gridsize=50, kernel='epak'):

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    tau = df.tau.iloc[0]
    M_min, M_max = min(df.M), max(df.M)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.M_std), max(df.M_std)
    M_std = np.linspace(M_std_min, M_std_max, num=num)

    x = M_std
    sig = np.zeros((num, 3))

    # TODO: speed up with tensor instead of loop
    for i, m in enumerate(x):
        sig[i] = smoothing_rookley(df, m, tau, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1] / np.std(df.M)
    second = sig[:, 2] / np.std(df.M)

    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, M_std
