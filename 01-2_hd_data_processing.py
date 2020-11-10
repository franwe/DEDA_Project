import os
from os.path import join
import pandas as pd
from datetime import datetime
from util.connect_db import connect_db, bulk_write

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
