import pymongo


class MongoDBDeserializer:
    _instance = None

    def __new__(cls, connection_string, database_name, collection_name):
        if cls._instance is None:
            cls._instance = super(MongoDBDeserializer, cls).__new__(cls)
            cls._instance.connection_string = connection_string
            cls._instance.database_name = database_name
            cls._instance.collection_name = collection_name
            cls._instance.deserialized_data = None
            cls._instance.client = pymongo.MongoClient(cls._instance.connection_string)
            cls._instance.database = cls._instance.client[cls._instance.database_name]
            cls._instance.collection = cls._instance.database[cls._instance.collection_name]
        return cls._instance

    def deserialize_data(self):
        result_cursor = self.collection.find({})
        self.deserialized_data = [document for document in result_cursor]

    def get_deserialized_data(self):
        if self.deserialized_data is None:
            self.deserialize_data()
        return self.deserialized_data

    def update_deserialized_data(self):
        current_data_length = self.collection.count_documents({})
        if current_data_length != len(self.deserialized_data):
            self.deserialize_data()
