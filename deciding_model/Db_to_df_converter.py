import pandas as pd

from db.MongoConnect import connect_to_collection
from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_new_train_collection_name_from_conf, get_number_of_highest_results_from_conf


def get_from_mongo_to_dataframe(new):
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_new_train_collection_name_from_conf())
    df = pd.DataFrame()
    if new:
        for doc in collection.find():
            for v in doc.items():
                if isinstance(v[1], dict):
                    # for items in v:
                    if is_dict_with_size(v[1], 17):
                        data = [[v[0]] + list(v[1].values())]
                        columns = get_unified_header(list(v[1]))
                        if df.empty:
                            df = pd.DataFrame(data, columns=columns)
                        else:
                            df_tmp = pd.DataFrame(data, columns=columns)
                            df = pd.concat([df, df_tmp], ignore_index=True)
                            df.reset_index()
    else:
        for doc in collection.find({}, {"_id": 0}):
            for v in doc.items():
                if is_dict_with_size(v[1], 12):
                    data = [[v[0]] + list(v[1].values())]
                    columns = get_unified_header(list(v[1]))
                    if df.empty:
                        df = pd.DataFrame(data, columns=columns)
                    else:
                        df_tmp = pd.DataFrame(data, columns=columns)
                        df = pd.concat([df, df_tmp], ignore_index=True)
                        df.reset_index()
    return df


def convert_to_df(data):
    models = ["clip"]
    _df = pd.DataFrame()
    k = (get_number_of_highest_results_from_conf() + 2) * len(models) + 3
    if isinstance(data, dict):
        for items in data.items():
            if is_dict_with_size(items[1], k):
                data = [[items[0]] + list(items[1].values())]
                columns = get_unified_header(list(items[1]))
                if _df.empty:
                    _df = pd.DataFrame(data, columns=columns)
                else:
                    df_tmp = pd.DataFrame(data, columns=columns)
                    _df = pd.concat([_df, df_tmp], ignore_index=True)
                    _df.reset_index()
    return _df


def is_dict_with_size(obj, size):
    return isinstance(obj, dict) and len(obj) == size


def get_unified_header(original_list):
    # Initialize counters for clip and resnet
    clip_count = 1
    resnet_count = 1
    orb_count = 1

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
            elif "orb" in item:
                new_item = f"orb{orb_count}"
                orb_count += 1
            else:
                new_item = item
        else:
            new_item = item
        new_list.append(new_item)

    id_element = 'id'
    return [id_element] + new_list
