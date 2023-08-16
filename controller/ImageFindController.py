from flask import request

from controller.BaseController import BaseController
from image_find import FrameHandler


class ImageFindController(BaseController):
    ENRICH_ROUTE = "/api/find"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}/image", methods=["POST"])
        def find_image():
            return FrameHandler.handle_request(request, self.config.get_config().get('common').get('upload_folder'))
