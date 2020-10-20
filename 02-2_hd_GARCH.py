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
RndData = RndDataClass(cutoff=0.4)
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
