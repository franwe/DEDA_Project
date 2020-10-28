import os
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from util.connect_db import connect_db, bulk_write, get_as_df
from datetime import datetime

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_plots = join(cwd, "plots") + os.sep
db = connect_db()


d_binance = pd.read_csv(source_data + "BTCUSDT_binance.csv", skiprows=1)
d_binance = d_binance[["Date", "Close"]]
d_binance.columns = ["date_str", "price"]
d_binance["datetime"] = d_binance.date_str.apply(
    lambda ds: datetime.strptime(ds, "%Y-%m-%d")
)
coll = db["BTCUSD_binance"]
bulk_write(coll, d_binance, ordered=False)

d_deribit = pd.read_csv(source_data + "BTCUSD_deribit.csv", sep=";")
d_deribit.columns = ["date_str", "price"]
d_deribit["datetime"] = d_deribit.date_str.apply(
    lambda ds: datetime.strptime(ds, "%Y-%m-%d")
)
coll = db["BTCUSD_deribit"]
bulk_write(coll, d_deribit, ordered=False)

# ------------------------------------------------------------------------ PLOT
query = {}
prices_deribit = get_as_df(db["BTCUSD_deribit"], query)
prices_deribit["time_dt"] = prices_deribit.datetime.apply(
    lambda ts: datetime.fromtimestamp(ts / 1000)
)


prices_binance = get_as_df(db["BTCUSD_binance"], query)
prices_binance["time_dt"] = prices_binance.datetime.apply(
    lambda ts: datetime.fromtimestamp(ts / 1000)
)

merged_prices = pd.merge(
    prices_deribit,
    prices_binance,
    how="left",
    on="time_dt",
    suffixes=("_deribit", "_binance"),
)

diff = merged_prices.price_deribit / merged_prices.price_binance
1 - diff.mean()
diff.std()

fig1 = plt.figure(figsize=(10, 5))
ax = fig1.add_subplot(111)
plt.xticks(rotation=45)

ax.plot(prices_binance.time_dt, prices_binance.price, "b", lw=1)
ax.scatter(merged_prices.time_dt, merged_prices.price_binance, s=4, c="b")
ax.scatter(merged_prices.time_dt, merged_prices.price_deribit, s=4, c="r")


locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.tight_layout()

fig1.savefig(join(save_plots, "BTCUSD_2016_2020.png"), transparent=True)
