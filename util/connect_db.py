from sshtunnel import SSHTunnelForwarder
import pymongo
import pprint

# ssh -i IRTG franziska_wehrmann@35.205.115.90

MONGO_HOST = '35.205.115.90'
MONGO_DB = 'cryptocurrency'
MONGO_COLLECTION = 'deribit_transactions'

server = SSHTunnelForwarder(
    MONGO_HOST,
    ssh_username='franziska_wehrmann',
    ssh_pkey='/Users/franziska/.ssh/IRTG',
    ssh_private_key_password='imdJSFZ#12',   # this is working, but somtimes timeout!
    remote_bind_address=('127.0.0.1', 27017)
)

server.start()

client = pymongo.MongoClient('127.0.0.1', server.local_bind_port)
db = client[MONGO_DB]
pprint.pprint(db.collection_names())

#
coll = db[MONGO_COLLECTION]

coll.find_one()

server.stop()