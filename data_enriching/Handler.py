import os

from tensorflow.python.lib.io.file_io import delete_file

from data_enriching.FeaturesExtractionService import run_model
from MongoConnect import connect_to_collection
from asyncio.log import logger


def handle_request(request, app_configs):
    try:
        images_list = save_images_from_request(request, app_configs)
        images_dto = run_models(request, images_list)
        save_to_db(images_dto)
        delete_saved_images(images_list)
        return saved_images_string(images_list)
    except Exception as e:
        logger.error('Failed to upload to ftp: ' + str(e))
    return "", 500


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
