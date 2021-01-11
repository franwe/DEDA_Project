from util.connect_db import connect_db, get_as_df


class RndDataClass:
    def __init__(self, cutoff):
        self.coll = connect_db()["trades_clean"]
        self.cutoff = cutoff

    def load_data(self, date_str):
        """Load all trades, with duplicates.
        It DOES make a difference in Fitting!"""
        x = self.cutoff
        query = {"date": date_str}
        d = get_as_df(self.coll, query)
        df = d[(d.M <= 1 + x) & (d.M > 1 - x)]
        df = df[(df.iv > 0.01)]
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

    def filter_data(self, date, tau_day, mode="unique"):
        self.load_data(date)
        if mode == "complete":
            filtered_by_date = self.complete
        elif mode == "unique":
            self.delete_duplicates()
            filtered_by_date = self.unique

        df_tau = filtered_by_date[(filtered_by_date.tau_day == tau_day)]
        df_tau = df_tau.reset_index()
        return df_tau


class HdDataClass:
    def __init__(self, target="price"):
        self.target = target
        self.coll = connect_db()["BTCUSD_binance"]
        self.complete = None
        self._load_data()

    def _load_data(self):
        """ Load complete BTCUSDT prices """
        prices_binance = get_as_df(self.coll, {})
        self.complete = prices_binance

    def filter_data(self, date):
        # yesterday = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
        # yesterday_str = str(yesterday.date())

        S0 = self.complete.loc[self.complete.date_str == date, "price"].iloc[0]
        df = self.complete[self.complete.date_str <= date]
        return df, S0
