import logging
from flask import Flask, request
from waitress import serve

from configuration.AppConfig import AppConfig
from data_enriching import Handler
from deciding_model import Model_Trainer
from image_find import FrameHandler


class MyFlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.config = AppConfig('configuration/app.config')
        self.configure_logging()
        self.setup_routes()

    def configure_logging(self):
        app_config = self.config.get_config().get('log')
        logging.basicConfig(level=app_config.get('level'),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def setup_routes(self):
        @self.app.route("/api/image/enrich", methods=["POST"])
        def process_image():
            return Handler.handle_request(request, self.config.get_config().get('common').get('upload_folder'))

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

    def run(self):
        app_config = self.config.get_config().get('waitress')
        logging.info("Starting Waitress server...")
        serve(self.app, host=app_config.get('host'), port=app_config.get('port'))


if __name__ == '__main__':
    my_app = MyFlaskApp()
    my_app.run()
