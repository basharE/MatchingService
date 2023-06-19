from asyncio.log import logger

from flask import Flask, request, jsonify
from numpy import load
import tensorflow as tf

from deciding_model import Db_to_df_converter, Model_Trainer
from image_find import FrameHandler
from data_enriching import Handler
from data_enriching.FeaturesServices import guess_final_place
from MongoConnect import connect_to_collection

UPLOAD_FOLDER = 'uploads'
root_dir = 'C:/Users/Bashar/Documents/Thesis/Thesis/classifier_data/data/latest' + str(1) + '/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/api/image/enrich", methods=["POST"])
def process_image():
    return Handler.handle_request(request, app.config['UPLOAD_FOLDER'])


@app.route("/api/image/find", methods=["POST"])
def find_image():
    return FrameHandler.handle_request(request, app.config['UPLOAD_FOLDER'])


@app.route("/api/data/train", methods=["GET"])
def train_model():
    Model_Trainer.train_model()
    return 'done', 200


@app.route("/api/v1/entry", methods=['POST'])
def insert():  # put application's code here
    """
       Function to create new users.
       """
    uri = "mongodb+srv://bashar:bashar@mymongo.xwi5zqs.mongodb.net/?retryWrites=true&w=majority"
    # Database and collection names
    database_name = "museum_data"
    collection_name = "images"
    try:
        # Create new users
        try:
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/json':
                body = request.json
            # body = ast.literal_eval(json.dumps(request.get_json()))
            print(body)
        except:
            # Bad request as request body is not available
            # Add message for debugging purpose
            return "", 400
        collection = connect_to_collection(uri, database_name, collection_name)
        mydict = {"name": "John", "address": "Highway 37"}

        record_created = collection.insert_one(mydict)
        resnet_processed_features = load(
            'C:/Users/Bashar/Documents/Thesis/Thesis/classifier_data/data/resnet_processed_features1.npy')
        clip_processed_features = load(
            'C:/Users/Bashar/Documents/Thesis/Thesis/classifier_data/data/clip_processed_features1.npy')

        guess_final_place(1, resnet_processed_features, clip_processed_features, get_generator())

        # Prepare the response
        if isinstance(record_created, list):
            # Return list of Id of the newly created item
            return jsonify([str(v) for v in record_created]), 201
        else:
            # Return Id of the newly created item
            return jsonify(str(record_created)), 201
    except Exception as e:
        logger.error('Failed to upload to ftp: ' + str(e))
        return "", 500


if __name__ == '__main__':
    app.run()


def get_generator():
    batch_size = 64
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    generator = datagen.flow_from_directory(
        root_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    return generator
