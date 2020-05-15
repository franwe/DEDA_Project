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

trades['datetime'] = trades.timestamp.apply(lambda ts:
                                            datetime.fromtimestamp(ts/1000.0))
trades['date'] = trades.datetime.apply(lambda d: datetime.date(d))

start = datetime(2020, 3, 9).date()
stop = datetime(2020, 3, 16).date()

trades_small = trades[(trades.date >= start) & (trades.date < stop)]
trades_small.to_csv(data_path + 'trades_1.csv')

# load smaller dataset, data cleaning from here
trades = pd.read_csv(data_path + 'trades_1.csv')

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

mean_part = trades.M - np.mean(trades.M)
std_part = np.std(trades.M)
trades['M_std'] = mean_part / std_part

calls = trades[trades.option == 'C'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                      'iv', 'M', 'M_std', 'r']]
puts = trades[trades.option == 'P'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                     'iv', 'M', 'M_std', 'r']]

calls.to_csv(data_path + 'calls_1.csv')
puts.to_csv(data_path + 'puts_1.csv')
