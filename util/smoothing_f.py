import numpy as np
from scipy.stats import norm

# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(M, m, h):
    u_m = (M-m)/h
    return norm.cdf(u_m)


def epanechnikov(M, m, h):
    u_m = (M-m)/h
    return 3/4 * (1-u_m)**2


def smoothing_rookley(df, m, h, kernel=gaussian_kernel):
    M = np.array(df.M)
    y = np.array(df.iv)
    n = df.shape[0]

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X = np.array([X1, X2, X3]).T

    ker = kernel(M, m, h)
    W = np.diag(ker)

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    return beta[0], beta[1], 2*beta[2]




def local_polynomial(df, h, gridsize=50, kernel='epak'):

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
        sig[i] = smoothing_rookley(df, m, h, kernel)

    smile = sig[:, 0]
    first = sig[:, 1]
    second = sig[:, 2]

    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K

# ------------------------------------------------------------------------ MAIN
import os
from matplotlib import pyplot as plt
from util.data import RndDataClass, HdDataClass
cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

x = 0.8
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
HdData = HdDataClass(data_path + 'BTCUSDT.csv')

df_tau = RndData.filter_data("2020-03-11", 9)

smile, first, second, M, S, K = local_polynomial(df_tau, h=0.1,
                                                 gridsize=140, kernel='epak')

plt.scatter(df_tau.M, df_tau.iv)
plt.plot(M, smile, 'r')
