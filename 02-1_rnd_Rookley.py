import os
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff?

RndData.analyse()

day = "2020-05-11"
print(RndData.analyse(day))
tau_day = 11

overwrite = False
reset_S = False

print(day, tau_day)
hd_data, S0 = HdData.filter_data(day)


df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="complete")
if reset_S:
    df_tau["S"] = S0
    df_tau["M"] = df_tau.S / df_tau.K
RND = RndCalculator(df_tau, tau_day, day, h_densfit=0.1)

X = np.array(RND.data.M)
y = np.array(RND.data.iv)
res_bandwidth, res_fit = RND.bandwidth_and_fit(X, y)
RND.h_m = res_bandwidth[0]
plt.plot(res_bandwidth[1], res_bandwidth[2])  # TODO: make dictionary
plt.show()
RND.smile, RND.first, RND.second, RND.M_smile = res_fit

RND.rookley()


plt.scatter(RND.data.M, RND.data.q_M)
plt.plot(RND.M, RND.q_M)
plt.vlines(1, 0, 3.5)
plt.show()

a = 1
# from util.density import integrate

# integrate(RND.M, RND.q_M)
# integrate(RND.K, RND.q_K)
