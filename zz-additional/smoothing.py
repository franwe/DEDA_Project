import numpy as np
from scipy.stats import norm

# B-Spline
import scipy.interpolate as interpolate

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
    M = np.array(df.M)
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


# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return norm.cdf(u_m) * norm.cdf(u_t)


def epanechnikov(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return 3/4 * (1-u_m)**2 * 3/4 * (1-u_t)**2


def local_polynomial(df, tau, h_m, h_t=0.05, gridsize=50, kernel='epak'):

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    M_min, M_max = min(df.M), max(df.M)
    M = np.linspace(M_min, M_max, gridsize)

    sig = np.zeros((num, 3))
    for i, m in enumerate(M):
        sig[i] = smoothing_rookley(df, m, tau, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1]
    second = sig[:, 2]

    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K


def bspline(M, smile, sections, degree=3):
    idx = np.linspace(0, len(M) - 1, sections+1, endpoint=True).round(0).astype('int')
    x = M[idx]
    y = smile[idx]

    t, c, k = interpolate.splrep(x, y, s=0, k=degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    pars = {'t': t, 'c': c, 'deg': k}
    points = {'x': x, 'y': y}
    return pars, spline, points


# ----------------------------------------------------------------- WITHOUT TAU

def gaussian_kernel_new(M, m, h_m):
    u_m = (M-m)/h_m
    return norm.cdf(u_m)


def epanechnikov_new(M, m, h_m):
    u_m = (M-m)/h_m
    return 3/4 * (1-u_m)**2


def smoothing_rookley_new(df, m, h_m, kernel=gaussian_kernel):
    M = np.array(df.M)
    y = np.array(df.iv)
    n = df.shape[0]

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X = np.array([X1, X2, X3]).T

    ker = kernel(M, m, h_m)
    W = np.diag(ker)

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    return beta[0], beta[1], 2*beta[2]




def local_polynomial_new(df, h_m, gridsize=50, kernel='epak'):

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    M_min, M_max = min(df.M), max(df.M)
    M = np.linspace(M_min, M_max, gridsize)

    sig = np.zeros((num, 3))
    for i, m in enumerate(M):
        sig[i] = smoothing_rookley(df, m, h_m, kernel)

    smile = sig[:, 0]
    first = sig[:, 1]
    second = sig[:, 2]

    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K