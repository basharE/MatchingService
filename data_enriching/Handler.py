import os

from tensorflow.python.lib.io.file_io import delete_file

from configuration.AppConfig import AppConfig
from data_enriching.FeaturesExtractionService import run_model
from MongoConnect import connect_to_collection
import logging

from images_selector.Orchestrator import orchestrate
from images_selector.PathUtils import create_path


def handle_image_request(request, app_configs):
    try:
        images_list = save_images_from_request(request, app_configs)
        images_dto = run_models(request, images_list)
        save_to_db(images_dto)
        delete_saved_images(images_list)
        return saved_images_string(images_list)
    except Exception as e:
        logging.error('Failed to upload to ftp: ' + str(e))
    return "", 500


def handle_video_request(request):
    try:
        top_images, images_paths = process_video(request)
        chosen_images = [images_paths[index] for index in top_images]
        images_dto = run_models(request, chosen_images)
        save_to_db(images_dto)
        delete_saved_images(top_images)
        return saved_images_string(top_images)
    except Exception as e:
        logging.error('Handle video request: ' + str(e))
    return "", 500


def get_directory_from_conf():
    conf = AppConfig('configuration/app.config')
    return conf.get_config().get('video').get('video_directory')


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
    uri = "mongodb+srv://bashar:bashar@mymongo.xwi5zqs.mongodb.net/?retryWrites=true&w=majority"
    # Database and collection names
    database_name = "museum_data"
    collection_name = "images"
    collection = connect_to_collection(uri, database_name, collection_name)
    collection.insert_one(image_data)


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
    return 'The images were saved to db successfully'
