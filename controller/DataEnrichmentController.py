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
            return Handler.handle_image_request(request, upload_folder)

        @self.app.route(f"{self.ENRICH_ROUTE}/video", methods=["POST"])
        def process_video():
            return Handler.handle_video_request(request)
