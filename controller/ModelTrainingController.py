import logging

from controller.BaseController import BaseController
from deciding_model import Model_Trainer


class ModelTrainingController(BaseController):
    ENRICH_ROUTE = "/api/train"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}", methods=["GET"])
        def train_model():
            logging.info("***** Starting Training Classification Model *****")
            Model_Trainer.train_model()
            logging.info("***** Training Classification Model Finished *****")
            return 'done', 200
