import os

from tensorflow.python.lib.io.file_io import delete_file

from configuration.ConfigurationService import get_directory_from_conf, get_database_uri_from_conf, \
    get_database_name_from_conf, get_database_collection_name_from_conf, get_frames_directory_from_conf, \
    get_image_directory_from_conf
from data_enriching.FeaturesExtractionService import FeatureExtractor
from data_enriching.TrainDataFrameBuilder import find_best_k_results
from db.MongoConnect import connect_to_collection
import logging

from db.MongoCustomQueries import insert_one_, save_as_new_train_data
from image_find.FrameHandler import extract_features, find_similarities, save_as_train_data
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
        cap = request.files['video']

        video_directory = get_directory_from_conf()
        video_name = cap.filename
        if video_name == '':
            return 'No selected video file'
        create_path(video_directory)
        cap.save(os.path.join(video_directory, cap.filename))  # Save the video file to the specified directory

        top_images, images_paths, video_name_without_ext, total_number_of_frames, table = orchestrate(video_name,
                                                                                                      "clip")
        chosen_images = [images_paths[index] for index in top_images]
        images_dto = run_clip(request, chosen_images)
        images_dto['clip_representative_images_number'] = len(top_images)
        images_dto['clip_representative_images'] = table
        images_dto['frames_number_clip'] = total_number_of_frames

        # top_images, images_paths, video_name_without_ext, total_number_of_frames, table = orchestrate(video_name,
        #                                                                                               "orb")
        # chosen_images = [images_paths[index] for index in top_images]
        # images_dto.update(run_orb(request, chosen_images))
        # images_dto['orb_representative_images_number'] = len(top_images)
        # images_dto['orb_representative_images'] = table
        # images_dto['frames_number_orb'] = total_number_of_frames

        # top_images, images_paths, video_name_without_ext, total_number_of_frames, table = orchestrate(video_name,
        #                                                                                               "resnet")
        # chosen_images = [images_paths[index] for index in top_images]
        # images_dto.update(run_resnet(request, chosen_images))
        # images_dto['resnet_representative_images_number'] = len(top_images)
        # images_dto['resnet_representative_images'] = table
        # images_dto['frames_number_resnet'] = total_number_of_frames

        images_dto['video_name'] = video_name_without_ext
        delete_saved_images(images_paths)
        try:
            delete_file(get_directory_from_conf() + cap.filename)
        except Exception as e:
            logging.warning(f"Error deleting {get_directory_from_conf() + video_name_without_ext}: {e}")
        os.rmdir(get_frames_directory_from_conf() + video_name_without_ext)
        save_to_db(images_dto)
        return saved_images_string(chosen_images)
    except Exception as e:
        logging.error('Handle video request: ' + str(e))
    return "", 500


def process_video(cap, model):
    video_directory = get_directory_from_conf()
    video_name = cap.filename
    if video_name == '':
        return 'No selected video file'
    create_path(video_directory)
    cap.save(os.path.join(video_directory, cap.filename))  # Save the video file to the specified directory
    return orchestrate(video_name, model)


def run_models(request, images_list):
    image_data = {'name': request.form['name'], 'description': request.form['description']}
    i = 1
    feature_extractor = FeatureExtractor()
    for image_path in images_list:
        clip_result = feature_extractor.run_model('clip', image_path)
        resnet_result = feature_extractor.run_model('resnet', image_path)
        image_data['image' + str(i)] = dict(image=image_path, clip=clip_result.tolist(), resnet=resnet_result.tolist())
        i = i + 1
    return image_data


def run_clip(request, images_list):
    image_data = {'name': request.form['name'], 'description': request.form['description']}
    i = 1
    feature_extractor = FeatureExtractor()
    for image_path in images_list:
        clip_result = feature_extractor.run_model('clip', image_path)
        image_data['clip_image' + str(i)] = dict(image=image_path, clip=clip_result.tolist())
        i = i + 1
    return image_data


def run_resnet(request, images_list):
    image_data = {'name': request.form['name'], 'description': request.form['description']}
    i = 1
    feature_extractor = FeatureExtractor()
    for image_path in images_list:
        resnet_result = feature_extractor.run_model('resnet', image_path)
        image_data['resnet_image' + str(i)] = dict(image=image_path, resnet=resnet_result.tolist())
        i = i + 1
    return image_data


def run_orb(request, images_list):
    image_data = {'name': request.form['name'], 'description': request.form['description']}
    i = 1
    feature_extractor = FeatureExtractor()
    for image_path in images_list:
        orb_result = feature_extractor.run_model('orb', image_path)
        image_data['orb_image' + str(i)] = dict(image=image_path, orb=orb_result.tolist())
        i = i + 1
    return image_data


def save_to_db(image_data):
    logging.info('Starting save_to_db')
    uri = get_database_uri_from_conf()
    database_name = get_database_name_from_conf()
    collection_name = get_database_collection_name_from_conf()
    collection = connect_to_collection(uri, database_name, collection_name)
    insert_one_(collection, image_data)
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
        try:
            delete_file(image_path)
        except Exception as e:
            logging.warning(f"Error deleting {image_path}: {e}")


def saved_images_string(images_list):
    return 'Processing video was done, the images were saved to db successfully. Images number: ' + str(
        len(images_list)), 200


def handle_labeling_request(request):
    image = request.files['image']
    class_of_image = request.form['class']
    image_features = extract_features(image, get_image_directory_from_conf())
    images_similarities = find_similarities(image_features, class_of_image)
    best_k_results = find_best_k_results(images_similarities)
    if class_of_image != 0:
        save_as_new_train_data(best_k_results)
        save_as_train_data(images_similarities)
    return 'best ', 200
