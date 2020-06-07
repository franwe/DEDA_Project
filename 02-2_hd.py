import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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
fig1 = plt.figure(figsize=(15, 8))
ax = fig1.add_subplot(111)

# sample = MC_return(d, target, tau_day, S0, M=1000)
sample, processes = SVCJ(tau_day, S0, M=1000, myseed=1)

# Use 3 different kernel to estimate
S = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    hd = density_estimation(sample, S, S0, h, kernel)
    ax.plot(S, hd, '-')

# Scatter plot of data samples and histogram
ax.scatter(sample, np.zeros(sample.shape[0]),
           zorder=15, color='red', marker='+', alpha=0.5, label='Samples')

# ---------------------------------------------------- SVCJ VS. RETURN BASED MC

sample_MC = MC_return(d, target, tau_day, S0, M)
sample_SVCJ, processes = SVCJ(tau_day, S0, M, myseed=1)

fig2 = plt.figure(figsize=(10, 6))
ax = fig2.add_subplot(111)

for name, sample in zip(['MC', 'SVCJ'], [sample_MC, sample_SVCJ]):
    S = np.linspace(sample.min()*0.99, sample.max()*1.01, num=500)
    hd = density_estimation(sample, S, S0, h, kernel='epanechnikov')
    ax.plot(S, hd, '-', label=name)
ax.legend()
