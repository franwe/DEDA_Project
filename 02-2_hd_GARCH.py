import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from util.data import HdDataClass, RndDataClass
from util.garch import get_returns, rolling_prediction, simulate_GARCH_moving, GARCH_fit, batch_GARCH_predict, batch_S
from util.historical_density import density_estimation

import sys
# sys.path.append('/Users/franziska/briedenCloud/Studium/6Semester/DEDA/DEDA_Project')

cwd = os.getcwd() + os.sep
source_data = os.path.join(cwd, 'data', '01-processed') + os.sep
save_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep

HdData = HdDataClass(source_data + 'BTCUSDT.csv')
hd_data = HdData.complete
log_returns = get_returns(hd_data, mode='log')*100


# --------------------------------- moving window over history, plot parameters
rolling_predictions, pars, bounds, ret_fit = rolling_prediction(log_returns,
                                                                tau_day=25,
                                                                burnin=100,
                                                                plot=True)

fig_pars, axes = plt.subplots(4,1, figsize=(8,6))

for i, name in zip(range(0,4), ['mu', 'omega', 'alpha', 'beta']):
    axes[i].plot(pars[:,i], label='arch.arch_model', c='b')
    # axes[i].plot(range(0, len(pars)), (pars[:, i] + 1.96*bounds[:, i]), ls=':', c='b')
    # axes[i].plot(range(0, len(pars)), (pars[:, i] - 1.96*bounds[:, i]), ls=':', c='b')
    axes[i].set_ylabel(name)
axes[0].legend()


# ---------------------------------------------------- moving window prediction
# ------------------------------------ in future Garch-De-Garch VS. Batch Garch
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=0.4)
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
day = '2020-04-03'
RndData.analyse(day)
M = 5000 # 5000

days = ['2020-03-20', '2020-04-03', '2020-03-06']
taus = [7,            7,            49]

days = ['2020-04-03']
taus = [7]

for day, tau_day in zip(days, taus):
    hd_data, S0 = HdData.filter_data(date=day)
    log_returns = get_returns(hd_data)*100

    filename = simulate_GARCH_moving(log_returns, S0, tau_day, day, M)

    S_sim = pd.read_csv(save_data + filename)
    sample = np.array(S_sim['S'])

    S = np.linspace(0.5*S0, 1.5*S0, num=100)
    hd_single = density_estimation(sample, S, h=0.1*S0)

    ################## BATCH
    model_fit, pars, bounds = GARCH_fit(log_returns[:-tau_day])
    returns_sim = batch_GARCH_predict(model_fit, tau_day, simulations=M)
    sample = batch_S(S0, returns_sim)
    hd_batch = density_estimation(sample, S, h=0.1*S0)

    filename = 'T-{}_{}_S-batch.pkl'.format(tau_day, day)
    pickle.dump(sample, open(save_data + filename, "wb"))

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(S/S0, hd_single, c='r', label='single')
    ax.plot(S/S0, hd_batch, c='b', label='batch')
    ax.legend(loc=2)
    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(save_plots + 'GARCH_T-{}_{}_singleVSbatch.png'.format(tau_day, day), transperent=True)




# ------------------------------------------------------------------- load data
from os.path import isfile, join

files = [f for f in os.listdir(save_data) if (isfile(join(save_data, f)) & (f.startswith('T-')) & (f.endswith('.csv')))]

M = np.linspace(0.5, 1.5, num=100)
for file in files:
    print(file)
    splits = file.split('_')
    tau_day = splits[0][2:]
    day = splits[1]

    hd_data, S0 = HdData.filter_data(date=day)

    filename = 'T-{}_{}_S-single.csv'.format(tau_day, day)
    S_sim = pd.read_csv(os.path.join(save_data, filename))
    sample_single = np.array(S_sim['S'])

    hd_single = density_estimation(sample_single/S0, M, h=0.1)

    ################## BATCH

    filename = 'T-{}_{}_S-batch.pkl'.format(tau_day, day)
    with open(os.path.join(save_data, filename), 'rb') as f:
        sample_batch = pickle.load(f)
    hd_batch = density_estimation(sample_batch / S0, M, h=0.1)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(M, hd_single, c='r', label='single')
    ax.plot(M, hd_batch, c='b', label='batch')
    ax.legend(loc=2)
    ax.text(0.99, 0.99, str(day) + '\n' + r'$\tau$ = ' + str(tau_day),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.set_xlabel('Moneyness')
    fig.tight_layout()
    fig.savefig(os.path.join(save_plots, 'GARCH_T-{}_{}_singleVSbatch.png'.format(tau_day, day)), transperent=True)


from util.historical_density import integrate
integrate(M, hd_single)

# ------------------------------------------------------------------- dax30.csv
# hd_data = hd_data[hd_data.Date < "2014-12-22"]
# S0 = hd_data[hd_data.Date == "2014-12-19"]['Adj.Close'].tolist()[0]
# log_returns = get_returns(hd_data)*100
#
# filename = 'HDdax_2014-12-22.csv'
# simulate_GARCH_moving(log_returns, S0, 25, M=100, filename=filename)
# S_sim = pd.read_csv(data_path + filename)
# sample = np.array(S_sim['S'])
#
# S = np.linspace(min(sample), max(sample), num=100)
# hd_single = density_estimation(sample, S, h=0.1*np.mean(sample))
#
# plt.plot(S/np.mean(sample), hd_single)
#
# from util.historical_density import sampling
#
# sample = sampling(data=hd_data, target='Adj.Close', tau_day=25, S0=np.mean(sample))
# hd_single = density_estimation(sample, S, h=0.1*np.mean(sample))
#
# plt.plot(S/np.mean(sample), hd_single)