

# --------------------
import numpy as np
import matplotlib.pyplot as plt
from localreg import *

num = 150
# K_q = np.linspace(Y_pred.min(), Y_pred.max(), num)
# M_q = np.linspace(1 - x, 1 + x, num)

a = df_tau.sort_values('M')
M_df = a.M.values
q_df = a.q.values

x, y = M_df, q_df
plt.plot(x, y, '+', markersize=0.6, color='gray')
y2 = localreg(x, y, degree=2, kernel=tricube, width=0.05)
plt.plot(x, y2, label='Local quadratic regression')
plt.legend()
plt.show()
plt.plot(S_hd/S0, hd)

given_value = 1
# a_list = x
absolute_difference_function = lambda list_value : abs(list_value - given_value)
closest_value = min(x, key=absolute_difference_function)
print(closest_value)

def Moneyness2K():
    """
    not tested yet, only tried a few things
    """
    df_tau[(df_tau.M > 0.995) & (df_tau.M <= 1.005)][['M', 'K', 'S', 'q', 'iv']]  # TODO: But Which S to choose?
    S_50 = df_tau[(df_tau.M > 0.995) & (df_tau.M <= 1.005)][['M', 'K', 'S', 'q', 'iv']].S.median()
    # TODO: --> for now S_50

    K_q = S0/M_df  # TODO: flip or not?  # S0 (hd) vs. S_50 (rnd)

    plt.plot(K_q, y2, '-', c='r')
    plt.plot(K_q, y, '.', markersize=2, color='gray')
    plt.plot(S_hd, hd, '-', c='b')


# -------------- HD 3D


tau_day = 2

for day in all_dates:
    hd_data, S0 = HdData.filter_data(date=day)
    fig3 = plot_2d(df_new, day, tau_day, hd_data, S0)
    hd, S_hd = simulate_hd(hd_data, S0, tau_day, S_domain=K_long)
    ax3.plot(M_long, hd, '-', c='b')

# --------------------------------------------------------------------- 3D PLOT
# TODO: Something wrong here?

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

d = RndData.complete
all_days = pd.date_range(start=d.date.min(), end=d.date.max())

color = cm.rainbow(np.linspace(0,1,len(all_days)))

fig3 = plt.figure(figsize=(7, 5))
ax3 = fig3.add_subplot(111, projection='3d')

y_pos = -1
i = 0

tau_day = 2
day0 = '2020-03-06'
all_days = ['2020-03-11', '2020-03-20', '2020-03-29', '2020-03-06']
color = cm.rainbow(np.linspace(0,1,len(all_days)))
for day in all_days:
    # day = str(day_ts.date())
    y_pos += 1

    c = color[i]
    i += 1

    # ------------------------- CALCULATIONS
    _hd_data, S0 = HdData.filter_data(date=day)
    hd_data, _S0 = HdData.filter_data(date=day0)

    S_hd = np.linspace((1-x) * S0, (1+x) * S0, num=100)
    hd, S_hd = simulate_hd(hd_data, S0, tau_day, S_domain=S_hd)
    y_adapted = [y_pos] * len(S_hd)
    ax3.plot(S_hd/S0, y_adapted, hd, c=c, ls='-')

plt.yticks(rotation=90)
new_locs = [i for i in range(0, len(all_days)) if i % 5 == 0]
new_labels = [str(day.date()) for i, day in
              zip(range(0, len(all_days)), all_days) if i % 5 == 0]
ax3.set_yticks(new_locs)
ax3.set_yticklabels(new_labels)

ax3.set_xlabel('Moneyness')
ax3.set_zlim(0)