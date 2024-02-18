import os
import numpy as np

from scipy.spatial.distance import euclidean

from data_enriching.FeaturesServices import get_similarity_
from db.MongoConnect import connect_to_collection
from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_images_collection_name_from_conf, get_number_of_highest_results_from_conf
from data_enriching.FeaturesExtractionService import FeatureExtractor
from db.MongoCustomQueries import save_as_train_data
import re

from db.MongoDBDeserializer import MongoDBDeserializer


def extract_features(image, app_configs):
    feature_extractor = FeatureExtractor()
    # saving image to tmp directory
    image_path = os.path.join(app_configs, image.filename)
    image.save(image_path)

    # extracting image features
    clip_result = feature_extractor.run_model('clip', image_path)
    # orb_result = feature_extractor.run_model('orb', image_path)
    orb_result = list()

    return dict(clip=clip_result, orb=orb_result, image_path=image_path)


def calculate_images_diff(image_name, tmp_doc, image_features):
    distances_dict = {}

    orb_db = tmp_doc.get("orb")
    orb_api = image_features.get("orb")
    if orb_db is not None and orb_api is not None:
        orb_db_ = np.array(orb_db, dtype='uint8')
        distances_dict[f"{image_name}_orb"] = get_similarity_(orb_db_, orb_api)

    clip_db = tmp_doc.get("clip")
    clip_api = image_features.get("clip")
    if clip_db is not None and clip_api is not None:
        distances_dict[f"{image_name}_clip"] = euclidean(clip_db[0], clip_api[0])

    resnet_db = tmp_doc.get("resnet")
    resnet_api = image_features.get("resnet")
    if resnet_db is not None and resnet_api is not None:
        distances_dict[f"{image_name}_resnet"] = euclidean(resnet_db[0], resnet_api[0])

    return distances_dict


def get_images_entries(doc, x):
    table_str_from_file = doc.get("clip_representative_images")
    lines = table_str_from_file.splitlines()
    data_rows = [line.split("|")[1:-1] for line in lines[3:-1]]

    # Filter out empty inner lists
    filtered_data_rows = [inner_list for inner_list in data_rows if bool(inner_list)]

    # Extract image names using regular expressions
    dic_keys = doc.keys()
    patterns = [r"clip_image\d+"]
    images_names_list = [entry for entry in dic_keys if any(re.findall(pattern, entry) for pattern in patterns)]

    # Determine the effective length based on x and the size of the filtered data
    length = max(x, len(filtered_data_rows))

    # Extract the first x elements from the images_names_list
    first_x_elements = images_names_list[:length]

    return first_x_elements


def calculate_distance(doc, image_features, class_of_image, x):
    images_entries = get_images_entries(doc, x)
    diff_dict = {}
    for image_name in images_entries:
        tmp_doc = doc.get(image_name)
        if tmp_doc is None:
            break
        diff_dict.update(calculate_images_diff(image_name, tmp_doc, image_features))

    diff_dict["class_of_image"] = class_of_image
    return diff_dict


def find_similarities(image_features, class_of_image):
    mongo_deserializer = MongoDBDeserializer(get_database_uri_from_conf(), get_database_name_from_conf(),
                                             get_database_images_collection_name_from_conf())

    deserialized_data = mongo_deserializer.get_deserialized_data()

    # Dictionary to store image comparisons
    images_comparison = {}
    x = get_number_of_highest_results_from_conf()
    # Iterate over documents in the collection
    for doc in deserialized_data:
        image_id = str(doc.get("_id"))
        distance = calculate_distance(doc, image_features, class_of_image == image_id, x)
        images_comparison[image_id] = distance
        image_name = str(doc.get("name"))
        distance["name"] = image_name
        image_description = str(doc.get("description"))
        distance["description"] = image_description

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


def handle_request(request, app_configs):
    image = request.files['image']
    class_of_image = request.form['class']
    image_features = extract_features(image, app_configs)
    images_similarities = find_similarities(image_features, class_of_image)
    if class_of_image != 0:
        save_as_train_data(images_similarities)
    return 'best ', 200
