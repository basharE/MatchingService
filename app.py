import logging
import os
from datetime import datetime

from flask import Flask
from waitress import serve

from configuration.AppConfig import AppConfig
from controller.DataEnrichmentController import DataEnrichmentController
from controller.HealthCheckController import HealthCheckController
from controller.ImageFindController import ImageFindController
from controller.ModelTrainingController import ModelTrainingController
from controller.TestFlowController import TestFlowController
from deciding_model.Model_Trainer import ClassifierTrainer


def train_before_starting():
    logging.info("***** Starting Train New Training Model *****")
    trainer_ = ClassifierTrainer()
    # trainer_.train_best_classifier(True)
    logging.info("***** Train New Training Model Finished *****")


class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.config = AppConfig('configuration/app.config')
        self.configure_logging()
        self.data_enrichment_controller = DataEnrichmentController(self.app, self.config)
        self.image_find_controller = ImageFindController(self.app, self.config)
        self.model_training_controller = ModelTrainingController(self.app, self.config)
        self.health_check_controller = HealthCheckController(self.app, self.config)
        self.test_flow_controller = TestFlowController(self.app, self.config)
        train_before_starting()

    def configure_logging(self):
        app_config = self.config.get_config().get('log')

        log_directory = app_config.get('log_directory')
        os.makedirs(log_directory, exist_ok=True)

        # Create a unique log file name with a timestamp
        log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = os.path.join(log_directory, log_file_name)

        logging.basicConfig(level=app_config.get('level'),
                            format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_file_path, mode='a')])

    def run(self):
        app_config = self.config.get_config().get('waitress')
        logging.info("Starting Waitress server...")
        serve(self.app, host=app_config.get('host'), port=app_config.get('port'))


if __name__ == '__main__':
    my_app = App()
    my_app.run()
