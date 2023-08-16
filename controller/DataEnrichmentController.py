from controller.BaseController import BaseController
from data_enriching import Handler
from flask import request


class DataEnrichmentController(BaseController):
    def setup_routes(self):
        @self.app.route("/api/image/enrich", methods=["POST"])
        def process_image():
            return Handler.handle_image_request(request, self.config.get_config().get('common').get('upload_folder'))

        @self.app.route("/api/video/enrich", methods=["POST"])
        def process_video():
            return Handler.handle_video_request(request)
