import numpy as np
import math


def expand_spline(spline, edge, degree, left_x, middle_x, right_x):
    p = np.polyfit(middle_x[:edge], spline[:edge], deg=degree)
    fct = np.poly1d(p)
    spline_left = fct(left_x)

    p = np.polyfit(middle_x[-edge:], spline[-edge:], deg=degree)
    fct = np.poly1d(p)
    spline_right = fct(right_x)

    spline_long = np.hstack([spline_left, spline, spline_right])
    return spline_long


def expand_X(X, add_left, add_right):
    gridsize = len(X)
    step_size = (X.max()-X.min())/(gridsize-1)
    left = np.arange(X.min()-step_size, 0,  -step_size)[:add_left][::-1]
    right = np.arange(X.max()+step_size, X.max()*3, step_size)[:add_right]
    X_long = np.hstack([left, X, right])
    return X_long, left, right


def expand(smile, first, second, M, S, K, edge=0.4):
    gridsize = len(M)
    edge = math.floor(gridsize*edge)

    # Moneyness
    M_long, M_left, M_right = expand_X(M, add_left=70, add_right=180)
    S_long, S_left, S_right = expand_X(S, add_left=70, add_right=180)
    K_long, K_left, K_right = expand_X(K, add_left=70, add_right=180)

    # first
    smile_long = expand_spline(spline=smile, edge=edge, degree=2,
                               left_x=M_left, middle_x=M, right_x=M_right)

    # first
    first_long = expand_spline(spline=first, edge=edge, degree=1,
                               left_x=M_left, middle_x=M, right_x=M_right)

    # second
    second_long = expand_spline(spline=second, edge=edge, degree=1,
                               left_x=M_left, middle_x=M, right_x=M_right)

    return smile_long, first_long, second_long, M_long, S_long, K_long