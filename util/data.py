import numpy as np
import pandas as pd


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
        d = d.drop('Unnamed: 0', axis=1)
        # d = d.drop_duplicates()
        print(d.shape)
        print('exclude values outside of {} - {} Moneyness - {}/{}'.format(1-x, 1+x,
                                    sum(d.M > 1+x) + sum(d.M <= 1-x), d.shape[0]))
        df = d[(d.M <= 1+x) & (d.M > 1-x)]
        print(df.shape)
        self.complete = df

    def analyse(self, date=None):
        if date is None:
            print(self.complete.date.value_counts())
        else:
            filtered_by_date = self.complete[(self.complete.date == date)]
            print(filtered_by_date.tau_day.value_counts())

    def delete_duplicates(self):
        self.unique = self.complete.drop_duplicates()

    def filter_data(self, date, tau_day, mode='complete'):
        if self.complete is None:
            print('load')
            self.load_data()

        if mode=='complete':
            filtered_by_date = self.complete[(self.complete.date == date)]
        elif mode=='unique':
            self.delete_duplicates()
            filtered_by_date = self.unique[(self.unique.date == date)]

        df_tau = filtered_by_date[(filtered_by_date.tau_day == tau_day)]
        df_tau = df_tau.reset_index()
        df_tau['M_std'] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
        return df_tau

