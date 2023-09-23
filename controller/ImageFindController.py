import logging

from flask import request, abort
from controller.BaseController import BaseController
from deciding_model.Model_Trainer import ClassifierTrainer
from services import ImageFindService
from services.ImageFindService import get_result_of_prediction


class ImageFindController(BaseController):
    ENRICH_ROUTE = "/api/find"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}/image", methods=["POST"])
        def find_image():
            logging.info("***** Starting Finding Image *****")

            trainer = ClassifierTrainer()
            best_classifier = trainer.best_classifier
            if best_classifier is None:
                # Raise a 404 Not Found exception
                abort(404, description="Best classifier not found, Please Train Data Before!")
            else:
                db_images_data_frame = ImageFindService.handle_request(request)
                response = get_result_of_prediction(trainer.predict_all(db_images_data_frame))

                logging.info("***** Finding Image Finished *****")
                return response
