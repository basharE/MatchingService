import logging

from controller.BaseController import BaseController
from deciding_model.Model_Trainer import ClassifierTrainer
from services import ModelTrainingService


class ModelTrainingController(BaseController):
    ENRICH_ROUTE = "/api/train"  # Define the common part of the route

    def setup_routes(self):
        trainer = ClassifierTrainer()

        @self.app.route(f"{self.ENRICH_ROUTE}", methods=["GET"])
        def train_model():
            logging.info("***** Starting Training Classification Model *****")
            trainer.train_best_classifier(False)
            logging.info("***** Training Classification Model Finished *****")
            return 'done', 200

        @self.app.route(f"{self.ENRICH_ROUTE}/build_new_train_data", methods=["GET"])
        def build_new_train_data():
            logging.info("***** Starting Building New Training Model *****")
            ModelTrainingService.build_new_train()
            logging.info("***** Building New Training Model Finished *****")
            return 'done', 200

        @self.app.route(f"{self.ENRICH_ROUTE}/new", methods=["GET"])
        def train_new_train_data():
            logging.info("***** Starting Train New Training Model *****")
            trainer.train_best_classifier(True)
            logging.info("***** Train New Training Model Finished *****")
            return 'done', 200

        @self.app.route(f"{self.ENRICH_ROUTE}/result", methods=["GET"])
        def train_result_data():
            ModelTrainingService.train_results()
            return 'done', 200
