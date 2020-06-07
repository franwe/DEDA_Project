import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import imageio

from util.historical_density import MC_return, SVCJ, density_estimation

cwd = os.getcwd() + os.sep


# ------------------------------------------------------------------- LOAD DATA
data_path = cwd + 'data' + os.sep
d = pd.read_csv(data_path + 'BTCUSDT.csv')

# -------------------------------------------------------------------- SETTINGS
target = 'Adj.Close'
tau_day = 10
M = 10000
h = 0.1
day = '2020-03-29'
S0 = d.loc[d.Date == day, target].iloc[0]
S = np.linspace(S0*0.3, S0*2, num=500)

# ------------------------------------------------------------ COMPUTE AND PLOT

# ----------------------------------------------------------- DIFFERENT KERNELS

tau_day = 3
h = 0.02

fig1 = plt.figure(figsize=(6, 4))
ax = fig1.add_subplot(111)

# sample = MC_return(d, target, tau_day, S0, M=1000)
sample, processes = SVCJ(tau_day, S0, n=1000, myseed=1)

# Use 3 different kernel to estimate
S = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    hd = density_estimation(sample, S, S0, h, kernel)
    ax.plot(S, hd, '-')

# Scatter plot of data samples and histogram
ax.scatter(sample, np.zeros(sample.shape[0]),
           zorder=15, color='red', marker='+', alpha=0.5, label='Samples')

# ---------------------------------------------------- SVCJ VS. RETURN BASED MC

def density_plot(tau_day):
    M = 10000
    h_MC = 0.1
    h_SVCJ = 0.02

    sample_MC = MC_return(d, target, tau_day, S0, M)
    # sample_SVCJ, processes = SVCJ(tau_day, S0, M, myseed=1)

    fig2 = plt.figure(figsize=(4, 4))
    ax = fig2.add_subplot(111)
    # for name, sample, h in zip(['MC', 'SVCJ'], [sample_MC, sample_SVCJ],
    #                            [h_MC, h_SVCJ]):

    for name, sample, h in zip(['MC'], [sample_MC],
                                   [h_MC]):
        S = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
        hd = density_estimation(sample, S, S0, h, kernel='epanechnikov')
        ax.plot(S, hd, '-', label=name)
    ax.set_xlim(1000, 12000)
    ax.set_ylim(0, 0.0011)
    ax.text(0.99, 0.99, r'$\tau$ = ' + str(tau_day),
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
    ax.set_xlabel('spot price')
    plt.tight_layout()

    # Used to return the plot as an image rray
    fig2.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    return image

kwargs_write = {'fps': 5.0, 'quantizer': 'nq'}
imageio.mimsave(data_path + 'HD_GIF' + os.sep + day + '_MC.gif',
                [density_plot(tau_day) for tau_day in range(1,100)], fps=5)

#    fig2.savefig(data_path + 'HD_GIF' + os.sep + day + '_' + str(tau_day).zfill(3) + '.png', transparent=True)

def plot_MC(tau_day):
    M = 10000
    h_MC = 0.1
    h_SVCJ = 0.02

    sample_MC = MC_return(d, target, tau_day, S0, M)
    # sample_SVCJ, processes = SVCJ(tau_day, S0, M, myseed=1)

    fig2 = plt.figure(figsize=(4, 4))
    ax = fig2.add_subplot(111)
    # for name, sample, h in zip(['MC', 'SVCJ'], [sample_MC, sample_SVCJ],
    #                            [h_MC, h_SVCJ]):

    for name, sample, h in zip(['MC'], [sample_MC],
                                   [h_MC]):
        S = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
        hd = density_estimation(sample, S, S0, h, kernel='epanechnikov')
        ax.plot(S, hd, '-', label=name)
    ax.set_xlim(1000, 12000)
    ax.set_ylim(0, 0.0011)
    ax.text(0.99, 0.99, r'$\tau$ = ' + str(tau_day),
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
    ax.set_xlabel('spot price')
    plt.tight_layout()
    return fig2

for tau_day in range(1,100):
    fig = plot_MC(tau_day)
    fig.savefig(data_path + 'HD_GIF' + os.sep + day + '_MC_' + str(tau_day).zfill(3) + '.png', transparent=True)
