import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from matplotlib.pyplot import cm

from util.historical_density import HdCalculator
from util.density import integrate
from util.data import HdDataClass, RndDataClass

cwd = os.getcwd() + os.sep
source_data = join(cwd, 'data', '01-processed') + os.sep
save_data = join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = join(cwd, 'plots') + os.sep
garch_data = join(cwd, 'data', '02-2_hd_GARCH') + os.sep



# ---------------------------------------------------- moving window prediction
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=0.4)
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
day = '2020-04-03'
RndData.analyse(day)
M = 5000 # 5000


def create_dates(start, end):
    dates = pd.date_range(start, end, closed='right', freq=pd.offsets.WeekOfMonth(week=1, weekday=2))
    return [str(date.date()) for date in dates]

days = create_dates(start='2019-04-01', end='2020-04-15')
taus = [7]*len(days)

color = cm.rainbow(np.linspace(0, 1, len(days)))
x_pos, y_pos = 0.99, 0.99
fig, ax = plt.subplots(1,1)
for day, tau_day, c in zip(days, taus, color):
    try:
        print(day, tau_day)
        hd_data, S0 = HdData.filter_data(day)
        HD = HdCalculator(hd_data, tau_day=tau_day, date=day,
                          S0=S0, burnin=tau_day * 2, path=garch_data, M=M,
                          overwrite=False)
        HD.get_hd()

        ax.plot(HD.M, HD.q_M, c=c)
        ax.text(x_pos, y_pos, str(day),
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes, c=c)
        y_pos -= 0.05
    except:
        print('Not enough historical data. Choose later timepoint.')
        pass

plt.xlabel('Moneyness M')
plt.xlim(0.5, 1.5)
plt.ylim(0)
plt.tight_layout()


integrate(HD.M, HD.q_M)
integrate(HD.K, HD.q_K)


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
    S_sim = pd.read_csv(join(save_data, filename))
    sample_single = np.array(S_sim['S'])

    hd_single = density_estimation(sample_single/S0, M, h=0.1)

    ################## BATCH

    filename = 'T-{}_{}_S-batch.pkl'.format(tau_day, day)
    with open(join(save_data, filename), 'rb') as f:
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
    fig.savefig(join(save_plots, 'GARCH_T-{}_{}_singleVSbatch.png'.format(tau_day, day)), transperent=True)


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