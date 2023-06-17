import os
import numpy as np

from scipy.spatial.distance import euclidean
from tensorflow.python.lib.io.file_io import delete_file

from MongoConnect import connect_to_collection
from data_enriching.FeaturesExtractionService import run_model


def extract_features(image, app_configs):
    # saving image to tmp directory
    image_path = os.path.join(app_configs, image.filename)
    image.save(image_path)

    # extracting image features
    clip_result = run_model('clip', image_path)
    resnet_result = run_model('resnet', image_path)

    delete_file(image_path)

    # return result
    return dict(clip=clip_result, resnet=resnet_result)


def calculate_images_diff(tmp_doc, image_features):
    distances_list = []
    resnet_db = np.array(tmp_doc.get("resnet"))
    resnet_api = image_features.get("resnet")
    if resnet_db is not None and resnet_api is not None:
        distances_list.append(euclidean(resnet_db[0], resnet_api[0]))

    clip_db = np.array(tmp_doc.get("clip"))
    clip_api = image_features.get("clip")
    if clip_db is not None and clip_api is not None:
        distances_list.append(euclidean(clip_db[0], clip_api[0]))

    return distances_list


def calculate_distance(doc, image_features):
    diff_list = []
    i = 1
    while i < len(doc):
        tmp_doc = doc.get("image" + str(i))
        if tmp_doc is not None:
            diff_list.extend(calculate_images_diff(tmp_doc, image_features))
            i += 1
        else:
            break
    return diff_list


def find_similarities(image_features):
    uri = "mongodb+srv://bashar:bashar@mymongo.xwi5zqs.mongodb.net/?retryWrites=true&w=majority"
    # Database and collection names
    database_name = "museum_data"
    collection_name = "images"
    collection = connect_to_collection(uri, database_name, collection_name)
    images_comparison = {}
    for doc in collection.find({}, {"_id": 1, "image1": 1, "image2": 1, "image3": 1, "image4": 1}):
        images_comparison[doc.get("_id")] = calculate_distance(doc, image_features)
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
    image_features = extract_features(image, app_configs)
    images_similarities = find_similarities(image_features)
    return 'best image is: ' + str(find_image_most_similarity(images_similarities)), 200
