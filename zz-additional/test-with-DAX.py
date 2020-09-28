import os
from matplotlib import pyplot as plt
from localreg import *

from util.data import RndDataClass, HdDataClass
from util.smoothing import local_polynomial
from util.smoothing import bspline
from util.risk_neutral_density_bu import spd_appfinance, spd_sfe

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep
HdData = HdDataClass(data_path + 'BTCUSDT.csv')
RndData = RndDataClass(data_path + 'trades_clean.csv', cutoff=0.5)

# -------------------------------     format SFE_RND_HD data to fit our example
import pandas as pd
dax30 = pd.read_csv(data_path + 'DAX30.csv')
hd_data = dax30[['Date', 'Adj Close']]
hd_data.columns = ['Date', 'Adj.Close']


odata = pd.read_csv(data_path + 'odata.txt', sep='\t')
odata['S_corrected'] = odata.spot * np.exp(-odata.rate * odata.maturity)
odata['M'] = odata.S_corrected/odata.strike
odata['maturity_day'] = round(odata.maturity * 365, 0)  # should be 255 no?
df = odata[['date', 'oprice', 'S_corrected', 'strike', 'maturity', 'maturity_day', 'ivola', 'M', 'rate']]
df.columns = ['date', 'P', 'S', 'K', 'tau', 'tau_day', 'iv', 'M', 'r']

df_tau = df[df.tau_day == 25]
day = "2014-12-22"
tau_day = 25
print(day)
# df_tau = RndData.filter_data(date=day, tau_day=tau_day, mode='complete')
# hd_data, S0 = HdData.filter_data(date=day)

h = df_tau.shape[0] ** (-1 / 9)
tau = df_tau.tau.iloc[0]
r = 0

fig3 = plt.figure(figsize=(5, 4))
ax3 = fig3.add_subplot(111)
# ------------------------------------------------------------------ SPD NORMAL
spd = spd_appfinance
spd = spd_sfe
smoothing_method = local_polynomial
X = np.array(df_tau.M)
Y = np.array(df_tau.iv)
smile, first, second, M, f = smoothing_method(X, Y, h)

# ---------------------------------------- B-SPLINE on SMILE, FIRST, SECOND
pars, spline, points = bspline(M, smile, sections=8, degree=3)
# derivatives
first_fct = spline.derivative(1)
second_fct = spline.derivative(2)

df_tau['q'] = df_tau.apply(lambda row: spd(row.M, row.S, row.K,
                                           spline(row.M), first_fct(row.M),
                                           second_fct(row.M),
                                           r, tau), axis=1)

a = df_tau.sort_values('M')
M_df = a.M.values
q_df = a.q.values
ax3.plot(M_df, q_df, '.', markersize=5, color='blue')

fit, first, second, X_domain, f = local_polynomial(M_df, q_df, h=0.08,
                                                   kernel='epak')
ax3.plot(X_domain, fit, '-', c='r')


plt.plot(X, Y)
plt.plot(M_df, q_df)
plt.scatter(df_tau.S/df_tau.M, df_tau.q)

# -------------------------------------------------------------------------- HD
import pandas as pd
from util.historical_density import density_estimation

filename = 'T-{}_{}_S-single.csv'.format(tau_day, day)
S_sim = pd.read_csv(data_path + filename)
sample = np.array(S_sim['S'])

S = np.linspace(0.5*S0, 1.5*S0, num=100)
hd_single = density_estimation(sample, S, h=0.1*S0)

plt.plot(S, hd_single)