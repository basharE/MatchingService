from data_enriching import Handler
from image_find import FrameHandler
from deciding_model import Model_Trainer
from flask import request
import logging


class Controller:
    def __init__(self, app, config):
        self.app = app
        self.config = config
        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/api/image/enrich", methods=["POST"])
        def process_image():
            return Handler.handle_image_request(request, self.config.get_config().get('common').get('upload_folder'))

        @self.app.route("/api/video/enrich", methods=["POST"])
        def process_video():
            return Handler.handle_video_request(request)

        @self.app.route("/api/image/find", methods=["POST"])
        def find_image():
            return FrameHandler.handle_request(request, self.config.get_config().get('common').get('upload_folder'))

        @self.app.route("/api/data/train", methods=["GET"])
        def train_model():
            Model_Trainer.train_model()
            return 'done', 200

        @self.app.route("/api/healthcheck", methods=["GET"])
        def health_check():
            logging.info("Server is alive...")
            return 'hello, Im alive', 200
