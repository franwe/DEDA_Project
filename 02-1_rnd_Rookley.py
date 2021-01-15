import os
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

from util.data import RndDataClass, HdDataClass
from util.risk_neutral_density import RndCalculator, plot_rookleyMethod
from util.smoothing import plot_locpoly_weights

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep

# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass()
RndData = RndDataClass(cutoff=x)
# TODO: Influence of coutoff? WHY IS THER ENO INFLUENCE WHY STILL MONEYNESS > 1.3? 2020-08-18 t=38

RndData.analyse()

day = "2020-05-12"
print(RndData.analyse(day))
tau_day = 45

overwrite = False
reset_S = False

print(day, tau_day)
hd_data, S0 = HdData.filter_data(day)

df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode="unique")
if reset_S:
    df_tau["S"] = S0
    df_tau["M"] = df_tau.S / df_tau.K
    df_tau = df_tau.drop_duplicates()
RND = RndCalculator(df_tau, tau_day, day)
print("Number of options today: ", RND.data.shape[0])

RND.fit_smile()

# X = np.array(RND.data.M)
# y = np.array(RND.data.iv)
# fig_weights = plot_locpoly_weights(
#     X, y, [0.8, 1, 1.25], h1=RND.h_m, h2=RND.h_m / 2
# )
# plt.show()
# plt.tight_layout()
# fig_weights.savefig(
#     join(save_plots, "Locpoly_Weights_{}.png".format(day)), transparent=True
# )

RND.rookley()

fig_method = plot_rookleyMethod(RND)
plt.tight_layout()
plt.show()
fig_method.savefig(
    join(save_plots, "RookleyMethod_{}.png".format(day)), transparent=True
)

a = 1
