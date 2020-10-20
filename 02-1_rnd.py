import os
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile, join
import matplotlib

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator
from util.historical_density import HdCalculator
from util.density import integrate
from util.data import HdDataClass, RndDataClass

cwd = os.getcwd() + os.sep
source_data = os.path.join(cwd, 'data', '00-raw') + os.sep
save_data = os.path.join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep
garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff?

RndData.analyse()

day = '2020-03-11'
RndData.analyse(day)
tau_day = 9

overwrite=False
reset_S = False

print(day, tau_day)
hd_data, S0 = HdData.filter_data(day)
HD = HdCalculator(hd_data, tau_day=tau_day, date=day,
                  S0=S0, burnin=tau_day * 2, path=garch_data, M=1000,
                  overwrite=overwrite)
HD.get_hd()

df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
if reset_S:
    df_tau['S'] = S0
    df_tau['M'] = df_tau.S / df_tau.K
RND = RndCalculator(df_tau, tau_day, day, h_densfit=0.1)
RND.fit_smile()
RND.rookley()




plt.scatter(RND.data.M, RND.data.q_M)
plt.plot(RND.M, RND.q_M)
plt.vlines(1, 0, 3.5)
plt.show()

from util.density import integrate
integrate(RND.M, RND.q_M)
integrate(RND.K, RND.q_K)
