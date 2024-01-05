from flask import request

from configuration.Decorators import log_request_and_response
from controller.BaseController import BaseController
from images_selector.Orchestrator import orchestrate
from services.TestFlowService import handle_single_request, handle_euclidean_request, handle_database_image, \
    handle_path_request


class TestFlowController(BaseController):
    ENRICH_ROUTE = "/api/test"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}/selector_model/image", methods=["GET"])
        @log_request_and_response
        def test_selector_model_image():
            handle_single_request(request)

        @self.app.route(f"{self.ENRICH_ROUTE}/selector_model/path", methods=["GET"])
        @log_request_and_response
        def test_selector_model_path():
            handle_path_request(request)

        """
            Calculate euclidean distance between two images,
            both will be accepted from request.
            Args:
                image1 (file): image from request.
                image2 (file): image from request.
            Returns:
                long: euclidean distance`.
        """

        @self.app.route(f"{self.ENRICH_ROUTE}/euclidean", methods=["GET"])
        @log_request_and_response
        def test_euclidean_calculation():
            return handle_euclidean_request(request)

        """
            Calculate euclidean distance between two images,
            one from request second existed in database.
            Args:
                image (file): image from request.
                image_name (string): text from request.
            Returns:
                long: euclidean distance`.
        """

        @self.app.route(f"{self.ENRICH_ROUTE}/dbimage", methods=["GET"])
        @log_request_and_response
        def test_database_image():
            return handle_database_image(request)

        @self.app.route(f"{self.ENRICH_ROUTE}/orch", methods=["GET"])
        @log_request_and_response
        def test_orchestrate():
            return orchestrate("20231117_152018.mp4", "clip")
