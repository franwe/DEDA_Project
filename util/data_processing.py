from datetime import datetime


def inst2date(s):
    expdate_str = s.split('-')[1]
    expdate = datetime.strptime(expdate_str, '%d%b%y').date()
    return expdate


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