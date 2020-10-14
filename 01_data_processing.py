import pandas as pd
import os
from datetime import datetime

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep


def inst2date(s):
    expdate_str = s.split('-')[1]
    expdate = datetime.strptime(expdate_str, '%d%b%y').date()
    return expdate


trades = pd.read_csv(data_path + 'Deribit_transactions_test.csv')
trades = trades.drop_duplicates()
trades = trades.reset_index()

trades['datetime'] = trades.timestamp.apply(lambda ts:
                                            datetime.fromtimestamp(ts/1000.0))
trades['date'] = trades.datetime.apply(lambda d: datetime.date(d))
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
trades.to_csv(data_path + 'trades_clean.csv', index=False)

calls = trades[trades.option == 'C'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                      'iv', 'M', 'r']]
puts = trades[trades.option == 'P'][['date', 'P', 'S', 'K', 'tau', 'tau_day',
                                     'iv', 'M', 'r']]


calls.to_csv(data_path + 'calls_clean.csv', index=False)
puts.to_csv(data_path + 'puts_clean.csv', index=False)


def data_processing(trades):
    trades['option'] = trades.instrument_name.apply(lambda s:
                                                    s.split('-')[3])
    trades['strike'] = trades.instrument_name.apply(lambda s:
                                                    float(s.split('-')[2]))
    trades['exp_date'] = trades.instrument_name.apply(lambda s: inst2date(s))

    trades['tau_day'] = trades.apply(lambda row:
                                     (row.exp_date - row.date).days, axis=1)
    trades['price_in_USD'] = trades.price * trades.index_price
    trades['iv'] = trades.iv.apply(lambda s: float(s) / 100)
    trades['p'] = trades.amount * trades.index_price

    # renaming and scaling for SPD
    trades.rename(
        columns={'strike': 'K', 'index_price': 'S', 'price_in_USD': 'P'},
        inplace=True)
    trades['r'] = 0
    trades['M'] = trades.S / trades.K
    trades['tau'] = trades.tau_day / 365

    trades = trades[
        ['date', 'P', 'S', 'K', 'tau', 'tau_day', 'iv', 'M', 'r', 'option',
         'direction']]
    return trades