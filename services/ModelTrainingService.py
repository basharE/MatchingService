from datetime import datetime
import pandas as pd

from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_train_collection_name_from_conf, get_database_new_train_collection_name_from_conf, \
    get_classifiers_collection_name_from_conf
from data_enriching.TrainDataFrameBuilder import find_best_k_results
from db.MongoConnect import connect_to_collection
from deciding_model.ResultTrainer import ResultTrainer


def build_new_train():
    collection_similarities = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                                    get_database_train_collection_name_from_conf())

    # Dictionary to store image comparisons
    images_new_training_data = list()

    # Iterate over documents in the collection
    for doc in collection_similarities.find():
        new_doc = get_only_dict(doc)
        images_new_training_data = images_new_training_data + list(find_best_k_results(new_doc).values())

    collection_train = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                             get_database_new_train_collection_name_from_conf())
    collection_train.delete_many({})
    current_datetime = datetime.now()

    # Convert datetime to string using strftime
    datetime_string = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    collection_train.insert_one({datetime_string: images_new_training_data})


def get_only_dict(doc):
    new_doc = {}
    for k, v in doc.items():
        if isinstance(v, dict):
            new_doc[k] = v
    return new_doc


def train_results():
    collection_results = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                               get_classifiers_collection_name_from_conf())

    columns = [str(i) for i in range(0, 63)]
    columns.append("class")

    df = pd.DataFrame(columns=columns)

    for doc in collection_results.find():
        # Convert the new data to a DataFrame with the same column structure
        df_to_add = pd.DataFrame([doc.get('data')], columns=df.columns)

        # Concatenate the new DataFrame to the existing DataFrame
        df = pd.concat([df, df_to_add], ignore_index=True)

    trainer = ResultTrainer()
    trainer.train_best_classifier(df)
