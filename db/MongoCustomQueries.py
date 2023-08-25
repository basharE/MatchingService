from datetime import datetime

from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_train_collection_name_from_conf, get_database_new_train_collection_name_from_conf
from db.MongoConnect import connect_to_collection


def insert_one_(collection, data):
    data['creation_date'] = datetime.now()

    # Insert the modified data document into the collection
    collection.insert_one(data)


def save_as_train_data(images_similarities):
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_train_collection_name_from_conf())
    insert_one_(collection, images_similarities)


def save_as_new_train_data(images_similarities):
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_new_train_collection_name_from_conf())
    insert_one_(collection, images_similarities)
