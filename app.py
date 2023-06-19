from flask import Flask, request
from deciding_model import Model_Trainer
from image_find import FrameHandler
from data_enriching import Handler

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


if __name__ == '__main__':
    app.run()
