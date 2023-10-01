import logging

from configuration.Decorators import log_request_and_response
from controller.BaseController import BaseController
from data_enriching import Handler
from flask import request


class DataEnrichmentController(BaseController):
    ENRICH_ROUTE = "/api/enrich"  # Define the common part of the route

    def setup_routes(self):
        common_config = self.config.get_config().get('common')
        upload_folder = common_config.get('upload_folder')

        @self.app.route(f"{self.ENRICH_ROUTE}/image", methods=["POST"])
        def process_image():
            logging.info("***** Starting Enriching Image *****")
            response = Handler.handle_image_request(request, upload_folder)
            logging.info("***** Enriching Image Finished *****")
            return response

        @self.app.route(f"{self.ENRICH_ROUTE}/video", methods=["POST"])
        @log_request_and_response
        def process_video():
            response = Handler.handle_video_request(request)
            return response

        @self.app.route(f"{self.ENRICH_ROUTE}/label", methods=["POST"])
        def label_image():
            logging.info("***** Starting Labeling Image *****")
            response = Handler.handle_labeling_request(request)
            logging.info("***** Labeling Image Finished *****")
            return response
