import pandas as pd

from db.MongoConnect import connect_to_collection
from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_new_train_collection_name_from_conf


def get_from_mongo_to_dataframe(new):
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_new_train_collection_name_from_conf())
    df = pd.DataFrame()
    if new:
        for doc in collection.find():
            for v in doc.values():
                if isinstance(v, list):
                    for items in v:
                        if is_dict_with_size(items, 16):
                            data = [list(items.values())]
                            columns = get_unified_header(list(items))
                            if df.empty:
                                df = pd.DataFrame(data, columns=columns)
                            else:
                                df_tmp = pd.DataFrame(data, columns=columns)
                                df = pd.concat([df, df_tmp], ignore_index=True)
                                df.reset_index()
    else:
        for doc in collection.find({}, {"_id": 0}):
            for v in doc.values():
                if is_dict_with_size(v, 12):
                    data = [list(v.values())]
                    columns = get_unified_header(list(v))
                    if df.empty:
                        df = pd.DataFrame(data, columns=columns)
                    else:
                        df_tmp = pd.DataFrame(data, columns=columns)
                        df = pd.concat([df, df_tmp], ignore_index=True)
                        df.reset_index()
    return df


def is_dict_with_size(obj, size):
    return isinstance(obj, dict) and len(obj) == size


def get_unified_header(original_list):
    # Initialize counters for clip and resnet
    clip_count = 1
    resnet_count = 1

    # Apply replacements
    new_list = []

    for item in original_list:
        if "image" in item:
            if "clip" in item:
                new_item = f"clip{clip_count}"
                clip_count += 1
            elif "resnet" in item:
                new_item = f"resnet{resnet_count}"
                resnet_count += 1
            else:
                new_item = item
        else:
            new_item = item
        new_list.append(new_item)

    return new_list
