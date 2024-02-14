from datetime import datetime

from flask import jsonify

from configuration.ConfigurationService import get_image_directory_from_conf
from data_enriching.TrainDataFrameBuilder import find_best_k_results
from deciding_model.Db_to_df_converter import convert_to_df
from image_find.FrameHandler import extract_features, find_similarities
from image_find.ImageWrapper import ImageWrapper

import io
import base64
import logging
from PIL import Image


def handle_request(request):
    try:
        features = get_image_features(request)
        images_similarities = find_similarities(features, None)
        best_k_results = find_best_k_results(images_similarities)
        return convert_to_df(best_k_results)
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")


def handle_request_(request):
    try:
        logging.info(f"Received request for image: {request.files['image'].filename}")
        features = handle_request_of_optimized_image(request)
        images_similarities = find_similarities(features, None)
        best_k_results = find_best_k_results(images_similarities)
        return convert_to_df(best_k_results), features.get("image_path")
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")


def get_result_of_prediction(data):
    if data is None:
        return "Can't find you position\nPlease try again"
    if data == -1:
        return "Found more than one candidate, try again\n"
    return "Your position is near, " + str(data) + "\n"


def get_image_features_(request):
    image = request.files['image']
    image_features = extract_features(image, get_image_directory_from_conf())
    return image_features


def get_image_features(request):
    data = request.get_json()

    base64_image_data = data.get('imageData')
    real_image_size = data.get('realImageSize')
    compressed_image_size = data.get('compressedImageSize')
    width = data.get('width')
    height = data.get('height')

    decoded_bytes = base64.b64decode(base64_image_data)
    compressed_image_stream = io.BytesIO(decoded_bytes)
    decompressed_image = Image.open(compressed_image_stream)

    # Rotate the image
    rotated_image = decompressed_image.rotate(270)

    # # Display the image using matplotlib
    # plt.imshow(np.array(rotated_image))
    # plt.show()

    # Get the rotated image as bytes
    rotated_image_bytes = rotated_image.tobytes()

    # Read the rotated image into BytesIO
    rotated_image_stream = io.BytesIO(rotated_image_bytes)

    width_, height_ = rotated_image.size
    image_size_bytes = len(rotated_image.tobytes())
    decoded_len = len(decoded_bytes)

    logging.info(
        f"Image Size (in bytes), from response: {real_image_size} bytes, current: {image_size_bytes} bytes")
    logging.info(
        f"Compressed Image Size (in bytes), from response: {compressed_image_size} bytes, current: {decoded_len} bytes")
    logging.info(f"Image Width, from response: {width}, current: {width_}")
    logging.info(f"Image Height, from response: {height}, current: {height_}")

    image_wrapper = ImageWrapper(rotated_image, filename="input_image.jpg")
    return extract_features(image_wrapper, get_image_directory_from_conf())


def handle_request_of_optimized_image(request):
    try:
        # Get the image file from the request
        image_data = request.files['image'].read()

        # Convert the image data to a PIL Image
        image = Image.open(io.BytesIO(image_data))
        rotated_image = image

        # plt.imshow(np.array(rotated_image))
        # plt.show()

        # return jsonify({'message': 'Image received successfully.'}), 200
        base_filename = "input_image"
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime('%Y%m%d_%H%M%S')
        file_extension = ".jpg"
        new_image_file_name = f"{base_filename}_{timestamp}{file_extension}"
        image_wrapper = ImageWrapper(rotated_image, filename=new_image_file_name)
        return extract_features(image_wrapper, get_image_directory_from_conf())

    except Exception as e:
        return jsonify({'error': str(e)}), 500
