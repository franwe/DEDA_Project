import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from matplotlib.pyplot import cm

from util.data import HdDataClass, RndDataClass
from util.historical_density import HdCalculator

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "02-3_rnd_hd") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep


# ---------------------------------------------------- moving window prediction
RndData = RndDataClass(cutoff=0.4)
HdData = HdDataClass()
M = 5000  # 5000


def create_dates(start, end):
    dates = pd.date_range(
        start,
        end,
        closed="right",
        freq=pd.offsets.WeekOfMonth(week=1, weekday=2),
    )
    return [str(date.date()) for date in dates]


days = create_dates(start="2019-09-01", end="2020-10-01")
taus = [42] * len(days)

color = cm.rainbow(np.linspace(0, 1, len(days)))
x_pos, y_pos = 0.99, 0.99
fig, ax = plt.subplots(1, 1)
for day, tau_day, c in zip(days, taus, color):
    hd_data, S0 = HdData.filter_data(day)
    hd_data = hd_data.reset_index()

    HD = HdCalculator(
        data=hd_data,
        S0=S0,
        path=garch_data,
        tau_day=tau_day,
        date=day,
        n=400,
        M=5000,
        overwrite=False,
    )

    try:
        HD.get_hd(variate=True)
        ax.plot(HD.M, HD.q_M, c=c)
        ax.text(
            x_pos,
            y_pos,
            str(day),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            c=c,
        )
        y_pos -= 0.05
        del HD
    except ValueError as e:
        print("ValueError :  ", e)
    except np.linalg.LinAlgError as e:
        print("np.linalg.LinAlgError :  ", e)

filename = join(save_plots, "HD-{}.png".format(tau_day))
fig.savefig(filename, transparent=True)
print("save figure to: ", filename)
plt.show()
