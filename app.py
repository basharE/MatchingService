import logging
from flask import Flask
from waitress import serve

from configuration.AppConfig import AppConfig
from controller.DataEnrichmentController import DataEnrichmentController
from controller.HealthCheckController import HealthCheckController
from controller.ImageFindController import ImageFindController
from controller.ModelTrainingController import ModelTrainingController


class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.config = AppConfig('configuration/app.config')
        self.configure_logging()
        self.data_enrichment_controller = DataEnrichmentController(self.app, self.config)
        self.image_find_controller = ImageFindController(self.app, self.config)
        self.model_training_controller = ModelTrainingController(self.app, self.config)
        self.health_check_controller = HealthCheckController(self.app, self.config)

    def configure_logging(self):
        app_config = self.config.get_config().get('log')
        logging.basicConfig(level=app_config.get('level'),
                            format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def run(self):
        app_config = self.config.get_config().get('waitress')
        logging.info("Starting Waitress server...")
        serve(self.app, host=app_config.get('host'), port=app_config.get('port'))


if __name__ == '__main__':
    my_app = App()
    my_app.run()
