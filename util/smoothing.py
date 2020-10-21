import numpy as np
from scipy.stats import norm

# B-Spline
import scipy.interpolate as interpolate

# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(x, Xi, h):
    u = (x - Xi) / h
    return norm.pdf(u)


def epanechnikov(x, Xi, h):
    u = (x - Xi) / h
    indicator = np.where(abs(u) <= 1, 1, 0)
    k = 0.75 * (1 - u ** 2)
    return k * indicator


def smoothing_rookley(X, Y, x, h, kernel=gaussian_kernel):
    n = X.shape[0]

    X1 = np.ones(n)
    X2 = X - x
    X3 = (X - x) ** 2
    X_matrix = np.array([X1, X2, X3]).T

    K_hn = 1 / h * kernel(X, x, h)
    f_hn = 1 / n * sum(K_hn)
    W_hn = K_hn / f_hn

    W = np.diag(W_hn)

    XTW = np.dot(X_matrix.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X_matrix)).dot(XTW).dot(Y)

    return beta[0], beta[1], 2 * beta[2], f_hn


def local_polynomial(X, Y, h, gridsize=50, kernel="epak"):

    if kernel == "epak":
        kernel = epanechnikov
    elif kernel == "gauss":
        kernel = gaussian_kernel
    else:
        print("kernel not know, use epanechnikov")
        kernel = epanechnikov

    X_domain = np.linspace(min(X), max(X), gridsize)

    sig = np.zeros((gridsize, 4))
    for i, x in enumerate(X_domain):
        sig[i] = smoothing_rookley(X, Y, x, h, kernel)

    fit = sig[:, 0]
    first = sig[:, 1]
    second = sig[:, 2]
    f = sig[:, 3]

    # S_min, S_max = min(df.S), max(df.S)
    # K_min, K_max = min(df.K), max(df.K)
    # S = np.linspace(S_min, S_max, gridsize)
    # K = np.linspace(K_min, K_max, gridsize)

    return fit, first, second, X_domain, f


def bspline(x, y, sections, degree=3):
    idx = np.linspace(0, len(x) - 1, sections + 1, endpoint=True).round(0).astype("int")
    x = x[idx]
    y = y[idx]

    t, c, k = interpolate.splrep(x, y, s=0, k=degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    pars = {"t": t, "c": c, "deg": k}
    points = {"x": x, "y": y}
    return pars, spline, points


#
# # ------------------------------------------------------------------------ MAIN
# import os
# from matplotlib import pyplot as plt
# from util.data import RndDataClass, HdDataClass
# cwd = os.getcwd() + os.sep
# data_path = cwd + 'data' + os.sep
#
# x = 5
# RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
# HdData = HdDataClass(data_path + 'BTCUSDT.csv')
#
# RndData.analyse("2020-03-11")
# df_tau = RndData.filter_data("2020-03-11", 44)
#
# X = np.array(df_tau.M)
# Y = np.array(df_tau.iv)
# smile, first, second, X_domain, f = local_polynomial(X, Y, h=0.2,
#                                                  gridsize=140, kernel='epak')
#
# plt.scatter(X, Y)
# plt.plot(X_domain, smile, 'r')
#
# plt.plot(X_domain, f)

# ---------------------------------------------------------------- BANDWIDTH CV
