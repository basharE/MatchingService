from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def connect_to_collection(uri, database_name, collection_name):
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        # Access the specified database
        db = client[database_name]

        # Access the specified collection
        col = db[collection_name]

        return col

    except Exception as e:
        print(e)

