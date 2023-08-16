import os

from tensorflow.python.lib.io.file_io import delete_file

from configuration.ConfigurationService import get_directory_from_conf, get_database_uri_from_conf, \
    get_database_name_from_conf, get_database_collection_name_from_conf, get_frames_directory_from_conf
from data_enriching.FeaturesExtractionService import run_model
from db.MongoConnect import connect_to_collection
import logging

from images_selector.Orchestrator import orchestrate
from utils.PathUtils import create_path


def handle_image_request(request, app_configs):
    try:
        images_list = save_images_from_request(request, app_configs)
        images_dto = run_models(request, images_list)
        save_to_db(images_dto)
        delete_saved_images(images_list)
        return saved_images_string(images_list)
    except Exception as e:
        logging.error('Failed to handle image: ' + str(e))
    return "Failed to handle image", 500


def handle_video_request(request):
    try:
        top_images, images_paths, video_name_without_ext = process_video(request)
        chosen_images = [images_paths[index] for index in top_images]
        images_dto = run_models(request, chosen_images)
        delete_saved_images(images_paths)
        os.rmdir(get_frames_directory_from_conf() + video_name_without_ext)
        save_to_db(images_dto)
        return saved_images_string(chosen_images)
    except Exception as e:
        logging.error('Handle video request: ' + str(e))
    return "", 500


def process_video(request):
    video_directory = get_directory_from_conf()
    cap = request.files['video']
    video_name = cap.filename
    if video_name == '':
        return 'No selected video file'
    create_path(video_directory)
    cap.save(os.path.join(video_directory, cap.filename))  # Save the video file to the specified directory
    return orchestrate(video_name)


def run_models(request, images_list):
    image_data = {'name': request.form['name'], 'description': request.form['description']}
    i = 1
    for image_path in images_list:
        clip_result = run_model('clip', image_path)
        resnet_result = run_model('resnet', image_path)
        image_data['image' + str(i)] = dict(image=image_path, clip=clip_result.tolist(), resnet=resnet_result.tolist())
        i = i + 1
    return image_data


def save_to_db(image_data):
    logging.info('Starting save_to_db')
    uri = get_database_uri_from_conf()
    database_name = get_database_name_from_conf()
    collection_name = get_database_collection_name_from_conf()
    collection = connect_to_collection(uri, database_name, collection_name)
    collection.insert_one(image_data)
    logging.info('Completed processing save_to_db to collection: %s', collection_name)


def save_images_from_request(request, app_configs):
    images = [request.files['image1'], request.files['image2'], request.files['image3'], request.files['image4']]
    images_names = []
    for image in images:
        file_path = os.path.join(app_configs, image.filename)
        image.save(file_path)
        images_names.append(file_path)
    return images_names


def delete_saved_images(images_list):
    for image_path in images_list:
        delete_file(image_path)


def saved_images_string(images_list):
    return 'Processing video was done, the images were saved to db successfully. Images number: ' + str(
        len(images_list)), 200
