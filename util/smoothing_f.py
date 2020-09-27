import numpy as np
from scipy.stats import norm

# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(M, m, h):
    u_m = (m-M)/h
    return norm.pdf(u_m)


def epanechnikov(M, m, h):
    u = (m - M)/h
    indicator = np.where(abs(u)<= 1, 1, 0)
    k = 0.75 * (1-u**2)
    return k * indicator


def smoothing_rookley(df, m, h, kernel=gaussian_kernel):
    M = np.array(df.M)
    y = np.array(df.iv)
    n = df.shape[0]

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X = np.array([X1, X2, X3]).T

    K_hn = 1/h * kernel(M, m, h)
    f_hn = 1/n * sum(K_hn)
    W_hn = K_hn/f_hn

    W = np.diag(W_hn)

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    return beta[0], beta[1], 2*beta[2], f_hn




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

    sig = np.zeros((num, 4))
    for i, m in enumerate(M):
        sig[i] = smoothing_rookley(df, m, h, kernel)

    smile = sig[:, 0]
    first = sig[:, 1]
    second = sig[:, 2]
    f = sig[:, 3]

    S_min, S_max = min(df.S), max(df.S)
    K_min, K_max = min(df.K), max(df.K)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, f

# ------------------------------------------------------------------------ MAIN
import os
from matplotlib import pyplot as plt
from util.data import RndDataClass, HdDataClass
cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

x = 5
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=x)
HdData = HdDataClass(data_path + 'BTCUSDT.csv')

RndData.analyse("2020-03-06")
df_tau = RndData.filter_data("2020-03-06", 7)

smile, first, second, M, S, K, f = local_polynomial(df_tau, h=0.2,
                                                 gridsize=140, kernel='epak')

plt.scatter(df_tau.M, df_tau.iv)
plt.plot(M, smile, 'r')

# ---------------------------------------------------------------- BANDWIDTH CV