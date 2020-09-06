import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RndDataClass:
    def __init__(self, path, cutoff=0.1):
        self.path = path
        self.cutoff = cutoff
        self.complete = None

        self._load_data()


    def _load_data(self):
        """ Load all trades, with duplicates. It DOES make a difference in Fitting! """
        x = self.cutoff
        d = pd.read_csv(self.path)
        print('Shape of raw data: ', d.shape)
        print('exclude values outside of {} - {} Moneyness - {}/{}'.format(1-x, 1+x,
                                    sum(d.M > 1+x) + sum(d.M <= 1-x), d.shape[0]))
        df = d[(d.M <= 1+x) & (d.M > 1-x)]
        print('Shape of limited Moneyness data: ', df.shape)
        self.complete = df

    def analyse(self, date=None):
        if date is None:
            print(self.complete.date.value_counts())
        else:
            filtered_by_date = self.complete[(self.complete.date == date)]
            print(filtered_by_date.tau_day.value_counts())

    def delete_duplicates(self):
        """
        Should I do it or not? It deletes if for same option was bought twice a day
        I guess better not delete, because might help for fitting to have weight
        of more trades to the "normal" values.
        """
        self.unique = self.complete.drop_duplicates()

    def filter_data(self, date, tau_day, mode='complete'):
        if mode=='complete':
            filtered_by_date = self.complete[(self.complete.date == date)]
        elif mode=='unique':
            self.delete_duplicates()
            filtered_by_date = self.unique[(self.unique.date == date)]

        df_tau = filtered_by_date[(filtered_by_date.tau_day == tau_day)]
        df_tau = df_tau.reset_index()
        df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
        return df_tau


class HdDataClass:
    def __init__(self, path, target='Adj.Close'):
        self.path = path
        self.target = target
        self.complete = None

        self._load_data()


    def _load_data(self):
        """ Load complete BTCUSDT prices """
        d = pd.read_csv(self.path)
        print('Shape of raw data: ', d.shape)
        print('from {} to {}'.format(d.Date.min(), d.Date.max()))
        self.complete = d

    def filter_data(self, date):
        yesterday = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)
        yesterday_str = str(yesterday.date())

        S0 = self.complete.loc[self.complete.Date == yesterday_str,
                               self.target].iloc[0]
        df_yesterday = self.complete[self.complete.Date <= yesterday_str]
        print('S0: ', S0)
        return df_yesterday, S0
