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
source_data = os.path.join(cwd, 'data', '01-processed') + os.sep
save_data = os.path.join(cwd, 'data', '02-3_rnd_hd') + os.sep
save_plots = os.path.join(cwd, 'plots') + os.sep
garch_data = os.path.join(cwd, 'data', '02-2_hd_GARCH') + os.sep


# ----------------------------------------------------------- LOAD DATA HD, RND
x = 0.5
HdData = HdDataClass(source_data + 'BTCUSDT.csv')
RndData = RndDataClass(source_data + 'trades_clean.csv', cutoff=x)


# ----------------------------------------------------------------- GARCH AGAIN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arch import arch_model
import arch.data.sp500


def get_log_returns(data, target):
    n = data.shape[0]
    data = data.reset_index()
    first = data.loc[:n - 2, target].reset_index()
    second = data.loc[1:, target].reset_index()
    historical_returns = (second / first)[target]
    log_returns = np.log(historical_returns) * 100
    return log_returns
    
data = arch.data.sp500.load().reset_index()

dates = data.reset_index().Date
months_ser = dates.apply(lambda dt: dt.month)
years_ser = dates.apply(lambda dt: dt.year)

months = dates.apply(lambda dt: dt.month).unique()
years = dates.apply(lambda dt: dt.year).unique()

df = pd.DataFrame()
for year in years:
    for month in months:
        mask = (months_ser == month) & (years_ser == year)
        first_day_of_month = data[mask].iloc[0]
        df = df.append(first_day_of_month, ignore_index=True)


log_returns = get_log_returns(df, 'Adj Close')

plt.plot(log_returns)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(log_returns**2)
plot_acf(abs(log_returns))
plot_acf(log_returns)
plot_pacf(log_returns**2)

data = log_returns/10
data = data - np.mean(data)
model = arch_model(data, p=1, q=1)
res = model.fit(disp='off')
print(res)







# -------------------------------------------------- S0 - don't take value from yesterday
import pandas as pd
from util.connect_db import connect_db, get_as_df
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_prices(prices, S0_today, S0_tomorrow):

    prices['time_dt'] = prices.datetime.apply(lambda ts: datetime.fromtimestamp(ts/1000))
    fig1 = plt.figure(figsize=(6, 3))
    ax = fig1.add_subplot(111)
    plt.xticks(rotation=45)

    ax.scatter(prices.time_dt, prices.index_price, 1)
    ax.text(0.99, 0.99, str(today),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    plt.axhline(S0_today, c='b', ls=':')
    plt.axhline(S0_tomorrow, c='r', ls=':')

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    return fig1

db = connect_db()
dates = pd.date_range('2020-03-04', '2020-04-22')
for start in dates:
    try:
        print(start)
        coll = db['BTCUSD']
        query = {'date': {'$et': str(start.date())}}
        query = {'date': str(start.date())}

        prices = get_as_df(coll, query)

        today = str(start.date())
        tomorrow = str((start + timedelta(days=1)).date())
        a, S0_today = HdData.filter_data(today)
        a, S0_tomorrow = HdData.filter_data(tomorrow)

        fig = plot_prices(prices, S0_today, S0_tomorrow)
        fig.savefig(save_plots + 'BTCUSD_{}.png'.format(today), transparent=True)
    except:
        print('no data ', start)




import pandas as pd
from datetime import datetime

today = '2020-03-12'
tomorrow = '2020-03-13'

df_all = pd.read_csv(join(cwd, 'data', '00-raw', 'Deribit_transactions_test.csv'))
df_all['datetime'] = df_all.timestamp.apply(lambda ts: datetime.fromtimestamp(ts/1000.0))

mask = df_all.datetime <= tomorrow
df_day_s = df_all[mask]
trades = df_day_s[df_day_s.datetime >= today]

trades['date'] = trades.datetime.apply(lambda d: datetime.date(d))
trades['time'] = trades.datetime.apply(lambda d: datetime.time(d))

prices = trades.groupby(by=['time', 'index_price']).count()
prices = prices.reset_index()
prices.to_csv('tmp-prices_{}.csv'.format(today))

# -------------------------------------------------- GAUSS DENSITY TRAFO K TO M
from util.density import density_trafo_K2M

def gauss_pdf(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)

S = 6000
sigma = 800
K = np.linspace(0.5*S, 1.5*S, num=200)
q_K = gauss_pdf(K, S, sigma)
M, q_M = density_trafo_K2M(K, q_K, S)

fig, axes = plt.subplots(2, 2, figsize=(7, 7))
# --------------------------------------------------- Moneyness - Moneyness
ax = axes[0,0]
ax.plot(K, gauss_pdf(K, mu=S, sigma=sigma))
ax.set_xlabel('K')
ax.title.set_text(r'$q_K(k)$')
ax.set_ylim(0)
ax.vlines(S, 0, 1)

ax = axes[1,0]
ax.plot(M, S/(M**2))
ax.set_xlabel('M')
ax.title.set_text(r'$abs(|J|) = S/M^2$')
ax.set_ylim(0)
ax.vlines(1, 0, (S/(M**2)).max()*1.1)

ax = axes[1,1]
ax.plot(M, gauss_pdf(S/M, mu=S, sigma=sigma))
ax.set_xlabel('M')
ax.set_ylim(0)
ax.title.set_text(r'$f_x(u(y)) = q_K(S/M)$')
# ax.set_xlim(0.5,1.5)
ax.vlines(1, 0, q_M.max())

ax = axes[0,1]
q = S/(M**2) * gauss_pdf(S/M, mu=S, sigma=sigma)
ax.plot(M, q)
ax.set_xlabel('M')
ax.title.set_text(r'$q_M(m) = S/m^2 \cdot q_K(S/m)$')
ax.set_ylim(0)
# ax.set_xlim(0.5,1.5)
ax.vlines(1, 0, q_M.max()*1.1)
plt.tight_layout()