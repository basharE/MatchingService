import logging

from flask import request, abort
from controller.BaseController import BaseController
from deciding_model.Model_Trainer import ClassifierTrainer
from services import ImageFindService


class ImageFindController(BaseController):
    ENRICH_ROUTE = "/api/find"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}/image", methods=["POST"])
        def find_image():
            logging.info("***** Starting Finding Image *****")
            response = ImageFindService.handle_request(request)


            trainer = ClassifierTrainer()
            best_classifier = trainer.best_classifier
            if best_classifier is None:
                # Raise a 404 Not Found exception
                abort(404, description="Best classifier not found, Please Train Data Before!")
            else:
                response = ImageFindService.handle_request(request)
                logging.info("***** Finding Image Finished *****")
                return response
