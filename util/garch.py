import os
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import time
from arch import arch_model

cwd = os.getcwd() + os.sep

def get_returns(data, target='Adj.Close', dt=1, mode='log'):
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - dt - 1, target].reset_index()
    second = data.loc[dt:, target].reset_index()
    historical_returns = (second / first)[target]
    if mode=='log':
        return np.log(historical_returns)
    elif mode=='linear':
        return historical_returns


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
    save_data = os.path.join(cwd, 'data', '02-2_hd_GARCH')
    tick = time.time()
    if filename is None: filename = 'T-{}_{}_S-single.csv'.format(tau_day, day)

    S_string = ""
    with open(save_data + filename, 'w') as file:
        file.write("index,S\n")

    for i in range(M):
        if i%(M*0.1) == 0:
            print('{}/{} - runtime: {} min'.format(i, M, round((time.time()-tick)/60)))
            with open(save_data + filename, 'a') as file:
                file.write(S_string)
            S_string = ""
        rolling_predictions, pars, bounds, ret_fit = rolling_prediction(log_returns,
                                                                       tau_day,
                                                                       plot=False)
        S_i = S_path(S0, ret_fit, tau_day)
        new_row = "{},{}\n".format(i, round(S_i))
        S_string += new_row
    return filename



# ------------------------------------------------------------- via fix horizon
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