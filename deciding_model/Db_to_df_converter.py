import pandas as pd

from MongoConnect import connect_to_collection


def get_from_mongo_to_dataframe():
    uri = "mongodb+srv://bashar:bashar@mymongo.xwi5zqs.mongodb.net/?retryWrites=true&w=majority"
    # Database and collection names
    database_name = "museum_data"
    collection_name = "train_data"
    collection = connect_to_collection(uri, database_name, collection_name)
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
