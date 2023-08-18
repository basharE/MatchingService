import os

from os import listdir
from os.path import isfile, join

import cv2
from scipy.spatial.distance import euclidean
from tensorflow.python.lib.io.file_io import delete_file

from configuration.ConfigurationService import get_frames_directory_from_conf, get_directory_from_conf, \
    get_threshold_const_from_conf
from data_enriching.FeaturesExtractionService import FeatureExtractor
from utils.PathUtils import create_path
from utils.SortingUtils import alphanum_key
from utils.VideoUtils import split_images_stream
import logging

from utils.Common import remove_extension


def count_ones(image_comparison_res):
    return image_comparison_res.count(1)


def find_max_ones(results_matrix):
    max_ones = 0
    max_index = 0
    for x in range(len(results_matrix)):
        tmp_max = count_ones(results_matrix[x])
        if tmp_max > max_ones:
            max_ones = tmp_max
            max_index = x
    return max_index


def is_representation_covered(results_matrix):
    for x in range(len(results_matrix[0])):
        if results_matrix[0][x] != 2:
            return False
    return True


def turn_ones_to_tows(results_matrix, index):
    for x in range(len(results_matrix[index])):
        if results_matrix[index][x] == 1:
            for y in range(len(results_matrix)):
                results_matrix[y][x] = 2


def get_top_images(results_matrix):
    logging.info('Starting get_top_images from total number of images: %s', len(results_matrix))
    top_images_with_ones = list()
    while not is_representation_covered(results_matrix):
        index_ = find_max_ones(results_matrix)
        top_images_with_ones.append(index_)
        turn_ones_to_tows(results_matrix, index_)
    logging.info('Completed processing get_top_images, returns %s images from total of %s', len(top_images_with_ones),
                 len(results_matrix))
    return top_images_with_ones


def extract_features(images_location, feature_extractor):
    logging.info('Starting extract_features in location: %s', images_location)
    features_list = list()
    images_names_list = list()
    only_files = [f for f in listdir(images_location) if isfile(join(images_location, f))]
    only_files.sort(key=alphanum_key)
    for x in range(len(only_files)):
        route = images_location + "/" + only_files[x]
        features_list.append(feature_extractor.run_model("clip", route))
        images_names_list.append(route)
    logging.info('Completed processing extract_features in location: %s', images_location)
    return features_list, images_names_list


def build_comparison_matrix(images_features):
    logging.info('Starting build_comparison_matrix')
    matrix = []
    for i in range(len(images_features)):
        row = []
        for j in range(len(images_features)):
            row.append(euclidean(images_features[i][0], images_features[j][0]))
        matrix.append(row)
    logging.info('Completed processing build_comparison_matrix with matrix size of: %s X %s', len(images_features),
                 len(images_features))
    return matrix


def convert_matrix_to_ones_zeros(results_matrix, threshold):
    logging.info('Starting convert_matrix_to_ones_zeros')
    for i in range(len(results_matrix)):
        for j in range(len(results_matrix[i])):
            if results_matrix[i][j] >= float(threshold):
                results_matrix[i][j] = 0
            else:
                results_matrix[i][j] = 1
    logging.info('Completed processing convert_matrix_to_ones_zeros with threshold of: %s', threshold)


def prepare_video(video_name):
    logging.info('Starting prepare_video: %s', video_name)
    if not os.path.exists(get_frames_directory_from_conf() + video_name):
        create_path(get_frames_directory_from_conf())
        logging.info('Starting orchestrate for file: %s', video_name)
        vid_cap = cv2.VideoCapture(get_directory_from_conf() + video_name)
        split_images_stream(vid_cap, video_name)
        try:
            delete_file(get_directory_from_conf() + video_name)
        except Exception as e:
            logging.warning(f"Error deleting {get_directory_from_conf() + video_name}: {e}")
        logging.info('Completed processing prepare_video: %s', video_name)
    else:
        logging.info('No need to process prepare_video: %s, image set already exist', video_name)


def orchestrate(video_name):
    prepare_video(video_name)
    feature_extractor = FeatureExtractor()
    features_list, images_names = extract_features(get_frames_directory_from_conf() + remove_extension(video_name),
                                                   feature_extractor)
    results_matrix = build_comparison_matrix(features_list)
    convert_matrix_to_ones_zeros(results_matrix, get_threshold_const_from_conf())
    top_images = get_top_images(results_matrix)
    logging.info('Completed processing orchestrate for file: %s, images: %s', video_name, top_images)
    return top_images, images_names, remove_extension(video_name)
