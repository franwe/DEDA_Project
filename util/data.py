import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from util.connect_db import connect_db, get_as_df


class RndDataClass:
    def __init__(self, cutoff=0.5):
        self.coll = connect_db()["trades_clean"]
        self.cutoff = cutoff

    def load_data(self, date_str):
        """Load all trades, with duplicates.
        It DOES make a difference in Fitting!"""
        x = self.cutoff
        query = {"date": date_str}
        d = get_as_df(self.coll, query)
        # print('Shape of raw data: ', d.shape)
        # print('exclude values outside of {} - {} Moneyness - {}/{}'
        #       .format(1-x, 1+x, sum(d.M > 1+x) + sum(d.M <= 1-x), d.shape[0]))
        df = d[(d.M <= 1 + x) & (d.M > 1 - x)]
        # print('Shape of limited Moneyness data: ', df.shape)
        self.complete = df

    def analyse(self, date=None, sortby="date"):
        if date is None:
            cursor = self.coll.aggregate(
                [
                    {"$group": {"_id": "$date", "count": {"$sum": 1}}},
                    {"$sort": {"_id": -1}},
                ]
            )
        else:
            cursor = self.coll.aggregate(
                [
                    {"$match": {"date": date}},
                    {"$group": {"_id": "$tau_day", "count": {"$sum": 1}}},
                    {"$sort": {"_id": 1}},
                ]
            )
        return list(cursor)

    def delete_duplicates(self):
        """
        Should I do it or not? It deletes if for same option was bought twice a
        day. I guess better not delete, because might help for fitting to have
        weight of more trades to the "normal" values.
        """
        self.unique = self.complete.drop_duplicates()

    def filter_data(self, date, tau_day, mode="complete"):
        self.load_data(date)
        if mode == "complete":
            filtered_by_date = self.complete
        elif mode == "unique":
            self.delete_duplicates()
            filtered_by_date = self.unique

        df_tau = filtered_by_date[(filtered_by_date.tau_day == tau_day)]
        df_tau = df_tau.reset_index()
        df_tau["M_std"] = (df_tau.M - np.mean(df_tau.M)) / np.std(df_tau.M)
        return df_tau


class HdDataClass:
    def __init__(self, path, target="Adj.Close"):
        self.path = path
        self.target = target
        self.complete = None

        self._load_data()

    def _load_data(self):
        """ Load complete BTCUSDT prices """
        d = pd.read_csv(self.path)
        # print('Shape of raw data: ', d.shape)
        # print('from {} to {}'.format(d.Date.min(), d.Date.max()))
        self.complete = d

    def filter_data(self, date):
        yesterday = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
        yesterday_str = str(yesterday.date())

        S0 = self.complete.loc[self.complete.Date == yesterday_str, self.target].iloc[0]
        df_yesterday = self.complete[self.complete.Date <= yesterday_str]
        return df_yesterday, S0
