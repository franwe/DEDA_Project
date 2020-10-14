import os
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from util.data import HdDataClass, RndDataClass
from util.garch import get_returns
from util.historical_density import density_estimation

import sys, os
# sys.path.append('/Users/franziska/briedenCloud/Studium/6Semester/DEDA/DEDA_Project')

cwd = os.getcwd() + os.sep
garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep


def GARCH_fit(data):
    model = arch_model(data, p=1, q=1)
    res2 = model.fit(disp='off')

    pars = [res2.params.mu, res2.params.omega, res2.params['alpha[1]'],
            res2.params[
                'beta[1]']]
    bounds = [res2.std_err.mu, res2.std_err.omega, res2.std_err['alpha[1]'],
              res2.std_err['beta[1]']]
    return res2, pars, bounds


def GARCH_predict(model_fit, mu, horizon=1):
    pred = model_fit.forecast(horizon=horizon)
    cond_std = np.sqrt(pred.variance.values[-1, :][0])
    simulated_ret = gauss(0, 1) * cond_std + mu
    predicted_var = np.sqrt(pred.variance.values[-1, :][0])
    return simulated_ret, predicted_var


def rolling_prediction(data, tau_day, burnin=None, window=None, plot=False):
    if not burnin:
        burnin = tau_day

    horizon = tau_day + burnin

    if window:
        inverseStart = horizon + window
    else:
        inverseStart = len(data)
        window = len(data) - horizon

    train = data[-inverseStart:-horizon]
    test = data[-horizon:]

    train = train.tolist()
    test = test.tolist()
    rolling_predictions = []
    pars_2 = np.zeros([len(test), 4])  # preallocate
    bounds_2 = np.zeros([len(test), 4])  # preallocate
    for i in range(horizon):
        train_window = train[-window:]

        model_fit, pars, bounds = GARCH_fit(train_window)
        bounds_2[i,:] = bounds
        pars_2[i, :] = pars

        simulated_ret, predicted_var = GARCH_predict(model_fit, mu=pars[0])
        train.append(simulated_ret)
        rolling_predictions.append(predicted_var)

    if plot:
        fig = plt.figure(figsize=(10,4))
        true, = plt.plot(test, label='true return')
        preds, = plt.plot(train[-horizon:], label='simulated return')
        plt.title('Simulated Return - Rolling Forecast')
        plt.legend()
    return rolling_predictions, pars_2, bounds_2, train[-horizon:]


def analyze(x):
    m = np.mean(x)
    s = np.std(x)
    print(m, s)
    return m, s


def S_path(S0, returns, tau_day, burnin=None):
    if not burnin:
        burnin = tau_day
    returns_trunc = returns[burnin:]
    returns_trunc = [i/100 for i in returns_trunc]
    return S0 * np.exp(sum(returns_trunc))


def simulate_GARCH_moving(log_returns, S0, tau_day, day=None, M=5000, filename=None):
    tick = time.time()
    if filename is None: filename = 'T-{}_{}_S-single.csv'.format(tau_day, day)

    S_string = ""
    with open(garch_data + filename, 'w') as file:
        file.write("index,S\n")

    for i in range(M+1):
        if i%(M*0.1) == 0:
            print('{}/{} - runtime: {} min'.format(i, M, round((time.time()-tick)/60)))
            with open(garch_data + filename, 'a') as file:
                file.write(S_string)
            S_string = ""
        rolling_predictions, pars, bounds, ret_fit = rolling_prediction(log_returns,
                                                                       tau_day,
                                                                       plot=False)
        S_i = S_path(S0, ret_fit, tau_day)
        new_row = "{},{}\n".format(i, round(S_i))
        S_string += new_row
    return filename

def M_path(returns, tau_day, burnin=None):
    if not burnin:
        burnin = tau_day
    returns_trunc = returns[burnin:]
    returns_trunc = [i/100 for i in returns_trunc]
    return np.exp(sum(returns_trunc))

#
# HdData = HdDataClass(data_path + 'BTCUSDT.csv')
# hd_data = HdData.complete
# log_returns = get_returns(hd_data, mode='log')*100
#
#
# rolling_predictions, pars, bounds, ret_fit = rolling_prediction(log_returns,
#                                                                 tau_day=25,
#                                                                 burnin=100,
#                                                                 plot=True)
#
# fig_pars, axes = plt.subplots(4,1, figsize=(8,6))
#
# for i, name in zip(range(0,4), ['mu', 'omega', 'alpha', 'beta']):
#     axes[i].plot(pars[:,i], label='arch.arch_model', c='b')
#     # axes[i].plot(range(0, len(pars)), (pars[:, i] + 1.96*bounds[:, i]), ls=':', c='b')
#     # axes[i].plot(range(0, len(pars)), (pars[:, i] - 1.96*bounds[:, i]), ls=':', c='b')
#     axes[i].set_ylabel(name)
# axes[0].legend()


# -------------------------------------------------------------- via horizon=40
def batch_GARCH_predict(model_fit, tau_day, simulations=10000, burnin=None):
    if not burnin:
        burnin = tau_day

    horizon = tau_day + burnin
    res = model_fit.forecast(horizon=horizon, method='simulation', simulations=simulations)
    returns_sim = res.simulations.values[-1,:,burnin:]
    return returns_sim


def batch_S(S0, returns_sim):
    a = np.dot(np.ones((1, returns_sim.shape[1])), returns_sim.T)
    S = S0 * np.exp(a/100)
    return S[0]


#--------------
RndData = RndDataClass(os.path.join(cwd, 'data', '01-processed', 'trades_clean.csv'), cutoff=0.4)
HdData = HdDataClass(os.path.join(cwd, 'data', '00-raw', 'BTCUSDT.csv'))
day = '2020-04-03'
RndData.analyse(day)
M = 5000 # 5000

days = ['2020-03-07', '2020-03-11', '2020-03-18', '2020-03-23', '2020-03-30', '2020-04-04']
taus = [2,2,2,2,2,2]


for day, tau_day in zip(days, taus):
    hd_data, S0 = HdData.filter_data(date=day)
    log_returns = get_returns(hd_data)*100

    filename = simulate_GARCH_moving(log_returns, S0, tau_day, day, M)

    S_sim = pd.read_csv(garch_data + filename)
    sample = np.array(S_sim['S'])

    S = np.linspace(0.5*S0, 1.5*S0, num=100)
    hd_single = density_estimation(sample, S, h=0.1*S0)

    ################## BATCH
    model_fit, pars, bounds = GARCH_fit(log_returns[:-tau_day])
    returns_sim = batch_GARCH_predict(model_fit, tau_day, simulations=M)
    sample = batch_S(S0, returns_sim)
    hd_batch = density_estimation(sample, S, h=0.1*S0)

    filename = 'T-{}_{}_S-batch.pkl'.format(tau_day, day)
    pickle.dump(sample, open(garch_data + filename, "wb"))

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