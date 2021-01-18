import numpy as np
from scipy.stats import norm
import scipy.interpolate as interpolate  # B-Spline
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import math
import random
import pandas as pd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(x, Xi, h):
    u = (x - Xi) / h
    return norm.pdf(u)


def epanechnikov(x, Xi, h):
    u = (x - Xi) / h
    indicator = np.where(abs(u) <= 1, 1, 0)
    k = 0.75 * (1 - u ** 2)
    return k * indicator


def local_polynomial_estimation(X, y, x, h, kernel):
    n = X.shape[0]
    K_i = 1 / h * kernel(x, X, h)
    f_i = 1 / n * sum(K_i)

    if f_i == 0:  # doesnt really happen, but in order to avoid possible errors
        W_hi = np.zeros(n)
    else:
        W_hi = K_i / f_i

    X1 = np.ones(n)
    X2 = X - x
    X3 = X2 ** 2

    X = np.array([X1, X2, X3]).T
    W = np.diag(W_hi)  # (n,n)

    XTW = (X.T).dot(W)  # (3,n)
    XTWX = XTW.dot(X)  # (3,3)
    XTWy = XTW.dot(y)  # (3,1)

    beta = np.linalg.pinv(XTWX).dot(XTWy)  # (3,1)
    return beta[0], beta[1], beta[2], W_hi


def linear_estimation(X, y, x, h, kernel):
    n = X.shape[0]

    K_i = 1 / h * kernel(x, X, h)

    f_i = 1 / n * sum(K_i)

    if f_i == 0:
        W_hi = np.zeros(n)
    else:
        W_hi = K_i / f_i

    y_pred = 1 / n * W_hi.dot(y)
    return y_pred, 0, 0, W_hi


def bandwidth_cv_singlerun(
    X,
    y,
    x_bandwidth,
    smoothing=local_polynomial_estimation,
    kernel=gaussian_kernel,
):
    n = X.shape[0]
    num = len(x_bandwidth)
    cv = np.zeros(num)
    for b, h in enumerate(x_bandwidth):
        sqrt_err = np.zeros(X.shape[0])
        for i, (x_test, y_test) in enumerate(zip(X, y)):
            X_temp, Y_temp = np.delete(X, i), np.delete(y, i)
            y_pred = smoothing(X_temp, Y_temp, x_test, h, kernel)[0]
            sqrt_err[i] = (y_test - y_pred) ** 2
        cv[b] = 1 / n * sum(sqrt_err)
    h = x_bandwidth[cv.argmin()]
    return (x_bandwidth, cv, h)


def bandwidth_cv_slicing(
    X,
    y,
    x_bandwidth,
    smoothing=local_polynomial_estimation,
    kernel=gaussian_kernel,
    no_slices=15,
):
    np.random.seed(1)
    df = pd.DataFrame(data=y, index=X)
    df = df.sort_index()
    X = np.array(df.index)
    y = np.array(df[0])
    n = X.shape[0]
    idx = list(range(0, n))
    slices = list(chunks(idx, math.ceil(n / no_slices)))
    if len(slices[0]) > 27:
        samples = 27
    else:
        samples = len(slices[0])

    num = len(x_bandwidth)
    cv = np.zeros(num)

    for b, h in enumerate(x_bandwidth):
        sqrt_err = np.zeros(no_slices)
        for i, chunk in enumerate(slices):
            X_train, X_test = np.delete(X, chunk), X[chunk]
            y_train, y_test = np.delete(y, chunk), y[chunk]

            runs = min(samples, len(chunk))
            sqrt_err_tmp = np.zeros(runs)
            for j, idx_test in enumerate(
                random.sample(list(range(0, len(chunk))), runs)
            ):
                y_pred = smoothing(
                    X_train, y_train, X_test[idx_test], h, kernel
                )[0]
                sqrt_err_tmp[j] = (y_test[idx_test] - y_pred) ** 2
            sqrt_err[i] = sum(sqrt_err_tmp)
        cv[b] = 1 / n * sum(sqrt_err)
    h = x_bandwidth[cv.argmin()]
    return (x_bandwidth, cv, h)


def bandwidth_cv(
    X,
    y,
    x_bandwidth,
    smoothing=local_polynomial_estimation,
    kernel=gaussian_kernel,
    show_plot=False,
):
    x_bandwidth_1, cv_1, h_1 = bandwidth_cv_slicing(X, y, x_bandwidth)

    stepsize = x_bandwidth[1] - x_bandwidth[0]
    x_bandwidth_2 = np.linspace(
        h_1 - (stepsize * 1.1), h_1 + (stepsize * 1.1), 10
    )

    x_bandwidth_2, cv_2, h_2 = bandwidth_cv_slicing(X, y, x_bandwidth_2)

    print(h_1, h_2)
    if show_plot:
        plt.plot(x_bandwidth_1, cv_1)
        plt.plot(x_bandwidth_2, cv_2)
        plt.show()
    return (x_bandwidth_2, cv_2, h_2), (x_bandwidth_1, cv_1, h_1)


def create_fit(
    X,
    y,
    h,
    gridsize=100,
    smoothing=local_polynomial_estimation,
    kernel=gaussian_kernel,
):
    X_domain = np.linspace(X.min(), X.max(), gridsize)
    fit = np.zeros(len(X_domain))
    first = np.zeros(len(X_domain))
    second = np.zeros(len(X_domain))
    for i, x in enumerate(X_domain):
        b0, b1, b2, W_hi = smoothing(X, y, x, h, kernel)
        fit[i] = b0
        first[i] = b1
        second[i] = b2
    return fit, first, second, X_domain


def plot_locpoly_weights(X, y, x_points, h1, h2, kernel=gaussian_kernel):
    import pandas as pd

    # this plot shows how the weights have different width for sparser data

    # ugly way to sort the values according to X (Moneyness)
    df = pd.DataFrame(data=y, index=X)
    df = df.sort_index()
    X = np.array(df.index)
    y = np.array(df[0])

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
        figsize=(4, 4),  # widht, hight
    )
    # density points
    y_density = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        y_density[i] = np.random.uniform(0, 1)

    ax1.scatter(X, y_density, c="k", alpha=0.5)
    ax1.tick_params(
        axis="y",  # changes apply to the y-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False,  # labels along the bottom edge are off)
    )
    ax1.set_xlabel("Moneyness")
    ax0.set_ylabel(r"Weight $W_i$")

    # weights
    for x, c in zip(x_points, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        b0, b1, b2, W_hi = local_polynomial_estimation(
            X, y, x, h1, kernel=kernel
        )
        b0, b1, b2, W_h = local_polynomial_estimation(
            X, y, x, h2, kernel=kernel
        )
        ax0.plot(X, W_hi, c=c)
        ax0.plot(X, W_h, ls=":", c=c)
        # print(sum(W_hi), sum(W_h))

    return fig


def bspline(x, y, sections, degree=3):
    idx = (
        np.linspace(0, len(x) - 1, sections + 1, endpoint=True)
        .round(0)
        .astype("int")
    )
    x = x[idx]
    y = y[idx]

    t, c, k = interpolate.splrep(x, y, s=0, k=degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    pars = {"t": t, "c": c, "deg": k}
    points = {"x": x, "y": y}
    return pars, spline, points


#
# # ---------------------------------------------------------------------- MAIN
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
