from sshtunnel import SSHTunnelForwarder
import pymongo
import pprint

# ssh -i IRTG franziska_wehrmann@35.205.115.90

MONGO_HOST = '35.205.115.90'
MONGO_DB = 'cryptocurrency'
MONGO_COLLECTION = 'deribit_transactions'

def connect_db(port=27017, db=MONGO_DB):
    client = pymongo.MongoClient(port=port)
    db = client[db]
    pprint.pprint(db.list_collection_names())
    return db

from datetime import datetime, timedelta
import pandas as pd
import json
from util.data_processing import data_processing



def get_as_df(collection, query):
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    return df
#
# db = connect_db()
# dates = pd.date_range('2020-03-04', '2020-04-22')
# for start in dates:
#     print(start)
#     end = start + timedelta(days=1) # datetime(2020,3,5)
#     start_ts = int(datetime.timestamp(start)*1000)
#     end_ts = int(datetime.timestamp(end)*1000)
#     coll = db[MONGO_COLLECTION]
#
#     query = {'$and': [{'timestamp': {'$gte': start_ts}},
#                       {'timestamp': {'$lte': end_ts}}]}
#     df_all = get_as_df(coll, query)
#
#     try:
#         df_all['datetime'] = df_all.timestamp.apply(lambda ts: datetime.fromtimestamp(ts/1000.0))
#         df_all['date'] = df_all.datetime.apply(lambda d: datetime.date(d))
#         df_all['time'] = df_all.datetime.apply(lambda d: datetime.time(d))
#
#
#         # ---------------------------------------------------------------- MERGE TRADES
#         coll = db['trades_clean']
#         last_id = coll.count_documents({})
#         columns = ['price', 'iv', 'instrument_name', 'index_price', 'direction', 'amount', 'date', 'time']
#         trades = df_all.groupby(by=columns).count().reset_index()[columns]
#         trades = data_processing(trades)
#         trades['_id'] = trades.index + last_id
#         records = json.loads(trades.T.to_json()).values()
#         coll.insert_many(records)
#
#         # --------------------------------------------------------------------- MERGE S
#         coll = db['BTCUSD']
#         columns = ['index_price', 'datetime', 'date', 'time']
#         S = df_all.groupby(by=columns).count().reset_index()[columns]
#         S['_id'] = S.apply(lambda row: str(row.date) + '_' + str(row.time), axis=1)
#         S['date'] = S.date.apply(lambda dt: str(dt))
#         # TODO: insert sort by TIME right now sorted by S, which is messy
#         records = json.loads(S.T.to_json()).values()
#         coll.insert_many(records)
#     except:
#
#         print('----- Date does not exist: ----- ', start.date())
#
#
# # -------------------------------------------------------------- SUM
# cursor = db.trades_clean.aggregate(
#    [{'$group' : {'_id': '$date', 'count' : {'$sum' : 1}}}]
# )
# pprint.pprint(list(cursor))

db = connect_db()


cursor = db.deribit_transactions.aggregate([
    { "$group": {
        '_id': None,
        "max": { "$max": "$timestamp" },
        "min": { "$min": "$timestamp" }
    }}
])
print(list(cursor))

print(datetime.fromtimestamp(1587424447310/1000),
      datetime.fromtimestamp(1583354259064/1000))

# Python program to find SHA256 hash string of a file
import hashlib

filename = '/Users/franziska/dump/cryptocurrency/deribit_transactions.bson'
sha256_hash = hashlib.sha256()
with open(filename, "rb") as f:
    # Read and update hash string value in blocks of 4K
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
    print(sha256_hash.hexdigest())
