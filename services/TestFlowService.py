import logging
import os
from os import listdir
from os.path import isfile, join

from PIL import Image
from flask import jsonify
from scipy.spatial.distance import euclidean

from configuration.ConfigurationService import get_image_directory_from_conf, get_database_uri_from_conf, \
    get_database_name_from_conf, get_database_images_collection_name_from_conf
from data_enriching.FeaturesServices import calculate_similarity
from db.MongoConnect import connect_to_collection
from image_find.FrameHandler import extract_features, find_similarities
from utils.PlotUtil import plot_line_graph
from utils.SortingUtils import alphanum_key


def get_for_graph(images_similarities):
    clip_values = list()
    resnet_values = list()
    object_indicator_for_clip = list()
    object_indicator_for_resnet = list()
    for k, v in images_similarities.items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                if "clip" in _k:
                    clip_values.append(_v)
                    object_indicator_for_clip.append(k)
                if "resnet" in _k:
                    resnet_values.append(_v)
                    object_indicator_for_resnet.append(k)
    return [i for i in range(len(clip_values))], clip_values, object_indicator_for_clip, [i for i in range(
        len(resnet_values))], resnet_values, object_indicator_for_resnet


def handle_path_request(request):
    response_headers = request.headers
    path = response_headers.get("path")
    images_files_list = get_images_list_from_path(path)
    for image_file in images_files_list:
        image_path = path + '/' + image_file
        image = read_image_from_path(image_path)
        image_identifier = get_image_identifier(image_path)
        handle(image, image_identifier)


def get_image_identifier(path):
    path_components = path.split('/')
    size = len(path_components)
    # Extract the desired parts
    if size >= 2:

        # Combine the 3rd and 4th components to get "20230719_093800/image1" (excluding the file extension)
        extracted_value = f"{path_components[size - 2]}/{os.path.splitext(path_components[size - 1])[0]}"
        output_string = extracted_value.replace('/', '_')

        return output_string
    else:
        print("Invalid file path")


def read_image_from_path(path):
    try:
        image = Image.open(path)
        # Now, 'image' contains the image data
        # You can perform various operations on the image using Pillow
        return image
    except FileNotFoundError:
        print(f"Image file not found: {path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_images_list_from_path(path):
    only_files = [f for f in listdir(path) if isfile(join(path, f))]
    only_files.sort(key=alphanum_key)
    return only_files


def handle_single_request(request):
    image = request.files['image']
    image_identifier = request.form['identity']
    handle(image, image_identifier)


def handle(image, image_identifier):
    image_features = extract_features(image, get_image_directory_from_conf())
    images_similarities = find_similarities(image_features, 0)

    clip_index_list, clip_similarity_list, class_clip, resnet_index_list, resnet_similarity_list, class_resnet = get_for_graph(
        images_similarities)

    # Plotting
    plot_line_graph(clip_index_list, clip_similarity_list, resnet_index_list, resnet_similarity_list,
                    (class_clip if len(class_clip) > len(class_resnet) else class_resnet),
                    "files/plot/" + image_identifier + "-graph.png", image_identifier)
    return ""


def handle_euclidean_request(request):
    image1 = request.files['image1']
    image2 = request.files['image2']
    image1_features = extract_features(image1, get_image_directory_from_conf())
    image2_features = extract_features(image2, get_image_directory_from_conf())
    image_path1 = os.path.join(get_image_directory_from_conf(), image1.filename)
    image_path2 = os.path.join(get_image_directory_from_conf(), image2.filename)

    image1_resnet_features = image1_features.get("resnet")
    image1_clip_features = image1_features.get("clip")

    image2_resnet_features = image2_features.get("resnet")
    image2_clip_features = image2_features.get("clip")

    res_orb = calculate_similarity(image_path1, image_path2)

    data = {
        "message": 'image1: ' + str(image1.filename) + ', image2: ' + str(image2.filename) + 'resnet_distance: ' + str(
            euclidean(image1_resnet_features[0], image2_resnet_features[0])) + ', clip_distance: ' + str(
            euclidean(image1_clip_features[0], image2_clip_features[0])) +
                   ', orb_distance: ' + str(res_orb)}
    logging.info(data)

    return jsonify(data)


def handle_database_image(request):
    resnet_res = ""
    clip_res = ""
    image = request.files['image']
    # image_name = request.files['image_name']
    image_name = "files/frames_streams/20230719_093800/image39.jpg"
    image_features = extract_features(image, get_image_directory_from_conf())

    # Connect to the MongoDB collection
    collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                       get_database_images_collection_name_from_conf())

    image1_resnet_features = image_features.get("resnet")
    image1_clip_features = image_features.get("clip")

    # Iterate over documents in the collection
    for doc in collection.find():
        for item in doc:
            if "clip" in item and isinstance(doc.get(item), dict) and doc.get(item).get("image") == image_name:
                clip_res = doc.get(item).get("clip")
            if "resnet" in item and isinstance(doc.get(item), dict) and doc.get(item).get("image") == image_name:
                resnet_res = doc.get(item).get("resnet")

    data = {"message": 'resnet_distance: ' + str(
        euclidean(image1_resnet_features[0], resnet_res[0])) + ', clip_distance: ' + str(
        euclidean(image1_clip_features[0], clip_res[0]))}
    return jsonify(data)
