import pandas as pd
import os
import numpy as np
from datetime import datetime

cwd = os.getcwd() + os.sep
data_path = cwd + 'Data' + os.sep


def inst2date(s):
    expdate_str = s.split('-')[1]
    expdate = datetime.strptime(expdate_str, '%d%b%y').date()
    return expdate


# load whole dataset, filter for start-stop date and save smaller dataset
trades = pd.read_csv(data_path + 'Deribit_transactions_test.csv')

# d = d.drop('Unnamed: 0', axis=1)  # TODO: Do this in my script as well?
trades = trades.drop('trade_id', axis=1)
trades = trades.drop_duplicates()

trades = trades[trades.iv > 0] # TODO: take out iv == 0 (and also 5? not so far)
trades['datetime'] = trades.timestamp.apply(lambda ts:
                                            datetime.fromtimestamp(ts/1000.0))
trades['date'] = trades.datetime.apply(lambda d: datetime.date(d))

# t = trades.groupby(by='date').mean()
# t.to_csv(data_path + 'BTC-prices.csv')

start = datetime(2020, 1, 23).date()
stop = datetime(2020, 3, 9).date()
# trades_small = trades[(trades.date >= start) & (trades.date < stop)]
# trades_small.to_csv(data_path + 'trades_0.csv')
#
# # load smaller dataset, data cleaning from here
# trades = pd.read_csv(data_path + 'trades_0.csv')

trades['option'] = trades.instrument_name.apply(lambda s:
                                                s.split('-')[3])
trades['strike'] = trades.instrument_name.apply(lambda s:
                                                float(s.split('-')[2]))
trades['exp_date'] = trades.instrument_name.apply(lambda s: inst2date(s))

trades['datetime'] = trades.timestamp.apply(lambda ts:
                                            datetime.fromtimestamp(ts/1000.0))
trades['date'] = trades.datetime.apply(lambda d:
                                       datetime.date(d))
trades['tau_day'] = trades.apply(lambda row:
                                 (row.exp_date - row.date).days, axis=1)

trades['price_in_USD'] = trades.price * trades.index_price
trades['iv'] = trades.iv.apply(lambda s: float(s)/100)


trades['p'] = trades.amount * trades.index_price

# renaming and scaling for SPD
trades.rename(columns={'strike': 'K', 'index_price': 'S', 'price_in_USD': 'P'},
              inplace=True)
trades['r'] = 0
trades['M'] = trades.S/trades.K
trades['tau'] = trades.tau_day / 365

trades = trades[['date', 'P', 'S', 'K', 'tau', 'tau_day', 'iv', 'M', 'r', 'option', 'direction']]
trades.to_csv(data_path + 'trades_clean.csv')

calls = trades[trades.option == 'C'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                      'iv', 'M', 'M_std', 'r']]
puts = trades[trades.option == 'P'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                     'iv', 'M', 'M_std', 'r']]


calls.to_csv(data_path + 'calls_0.csv')
puts.to_csv(data_path + 'puts_0.csv')
