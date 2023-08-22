import logging
import os
import numpy as np

from scipy.spatial.distance import euclidean
from tensorflow.python.lib.io.file_io import delete_file

from db.MongoConnect import connect_to_collection
from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_train_collection_name_from_conf, get_database_images_collection_name_from_conf
from data_enriching.FeaturesExtractionService import FeatureExtractor
from db.MongoCustomQueries import insert_one_


def extract_features(image, app_configs):
    feature_extractor = FeatureExtractor()
    # saving image to tmp directory
    image_path = os.path.join(app_configs, image.filename)
    image.save(image_path)

    # extracting image features
    clip_result = feature_extractor.run_model('clip', image_path)
    resnet_result = feature_extractor.run_model('resnet', image_path)

    try:
        delete_file(image_path)
    except Exception as e:
        logging.warning(f"Error deleting {image_path}: {e}")

    return dict(clip=clip_result, resnet=resnet_result)


def calculate_images_diff(image_name, tmp_doc, image_features):
    distances_dict = {}

    resnet_db = tmp_doc.get("resnet")
    resnet_api = image_features.get("resnet")
    if resnet_db is not None and resnet_api is not None:
        distances_dict[f"{image_name}_resnet"] = euclidean(resnet_db[0], resnet_api[0])

    clip_db = tmp_doc.get("clip")
    clip_api = image_features.get("clip")
    if clip_db is not None and clip_api is not None:
        distances_dict[f"{image_name}_clip"] = euclidean(clip_db[0], clip_api[0])

    return distances_dict


def get_images_entries(doc):
    images_names_list = list()
    dic_keys = doc.keys()
    for entry in dic_keys:
        if "image" in entry:
            images_names_list.append(entry)
    return images_names_list


def calculate_distance(doc, image_features, class_of_image):
    images_entries = get_images_entries(doc)
    diff_dict = {}
    for image_name in images_entries:
        tmp_doc = doc.get(image_name)
        if tmp_doc is None:
            break
        diff_dict.update(calculate_images_diff(image_name, tmp_doc, image_features))

    diff_dict["class_of_image"] = class_of_image
    return diff_dict


def find_similarities(image_features, class_of_image):
    # Connect to the MongoDB collection
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_images_collection_name_from_conf())

    # Dictionary to store image comparisons
    images_comparison = {}

    # Iterate over documents in the collection
    for doc in collection.find():
        image_id = str(doc.get("_id"))
        distance = calculate_distance(doc, image_features, class_of_image == image_id)
        distance.update({'zone': doc.get("description")})
        images_comparison[image_id] = distance

    return images_comparison


def list_average(lst):
    # average function
    avg = np.average(lst)
    return avg


def find_image_most_similarity(images_similarities):
    avg = 10
    key = ''
    for k, v in images_similarities.items():
        tmp_avg = list_average(v)
        if tmp_avg < avg:
            avg = tmp_avg
            key = k
    return key


def save_as_train_data(images_similarities):
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_train_collection_name_from_conf())
    insert_one_(collection, images_similarities)


def handle_request(request, app_configs):
    image = request.files['image']
    class_of_image = request.form['class']
    image_features = extract_features(image, app_configs)
    images_similarities = find_similarities(image_features, class_of_image)
    if class_of_image != 0:
        save_as_train_data(images_similarities)
    return 'best ', 200
