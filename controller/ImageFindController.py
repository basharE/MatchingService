from flask import request

from controller.BaseController import BaseController
from image_find import FrameHandler


class ImageFindController(BaseController):
    def setup_routes(self):
        @self.app.route("/api/image/find", methods=["POST"])
        def find_image():
            return FrameHandler.handle_request(request, self.config.get_config().get('common').get('upload_folder'))
