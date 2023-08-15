import pandas as pd

from MongoConnect import connect_to_collection
from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_train_collection_name_from_conf


def get_from_mongo_to_dataframe():
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_train_collection_name_from_conf())
    df = pd.DataFrame()
    for doc in collection.find({}, {"_id": 0}):
        for v in doc.values():
            data = [list(v.values())]
            columns = list(v)
            if df.empty:
                df = pd.DataFrame(data, columns=columns)
            else:
                df_tmp = pd.DataFrame(data, columns=columns)
                df = pd.concat([df, df_tmp], ignore_index=True)
                df.reset_index()
    return df
