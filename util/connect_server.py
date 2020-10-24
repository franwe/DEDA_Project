from sshtunnel import SSHTunnelForwarder
import pymongo
import pprint
from datetime import datetime

# ssh -i IRTG franziska_wehrmann@35.205.115.90

MONGO_HOST = "35.205.115.90"
MONGO_DB = "cryptocurrency"
MONGO_COLLECTION = "deribit_transactions"

server = SSHTunnelForwarder(
    MONGO_HOST,
    ssh_username="franziska_wehrmann",
    ssh_pkey="/Users/franziska/.ssh/IRTG",
    ssh_private_key_password="imdJSFZ#12",  # works, but if slow inter.timeout!
    remote_bind_address=("127.0.0.1", 27017),
)

server.start()

client = pymongo.MongoClient("127.0.0.1", server.local_bind_port)
db = client[MONGO_DB]
pprint.pprint(db.collection_names())

collection = db[MONGO_COLLECTION]

# coll.find_one()

cursor = collection.aggregate(
    [
        {
            "$group": {
                "_id": None,
                "max": {"$max": "$timestamp"},
                "min": {"$min": "$timestamp"},
            }
        }
    ]
)
print(list(cursor))

ts = [1583354259064, 1601122235251]
for t in ts:
    print(datetime.fromtimestamp(t / 1000))

server.stop()
