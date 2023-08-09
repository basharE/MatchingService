import logging
from flask import Flask, request
from deciding_model import Model_Trainer
from image_find import FrameHandler
from data_enriching import Handler

from waitress import serve

UPLOAD_FOLDER = 'uploads'
root_dir = 'C:/Users/Bashar/Documents/Thesis/Thesis/classifier_data/data/latest' + str(1) + '/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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


@app.route("/api/healthcheck", methods=["GET"])
def health_check():
    logging.info("Server is alive...")
    return 'hello, Im alive', 200


if __name__ == '__main__':
    logging.info("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=5000)
