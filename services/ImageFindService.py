from configuration.ConfigurationService import get_image_directory_from_conf
from data_enriching.TrainDataFrameBuilder import find_best_k_results
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
        print(best_k_results)
        logging.info("finished-----")
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")


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

    width_, height_ = decompressed_image.size
    image_size_bytes = len(decompressed_image.tobytes())
    decoded_len = len(decoded_bytes)

    logging.info(
        f"Image Size (in bytes), from response: {real_image_size} bytes, current: {image_size_bytes} bytes")
    logging.info(
        f"Compressed Image Size (in bytes), from response: {compressed_image_size} bytes, current: {decoded_len} bytes")
    logging.info(f"Image Width, from response: {width}, current: {width_}")
    logging.info(f"Image Height, from response: {height}, current: {height_}")

    image_wrapper = ImageWrapper(decoded_bytes, filename="input_image.jpg")
    return extract_features(image_wrapper, get_image_directory_from_conf())
