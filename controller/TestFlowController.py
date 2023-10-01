from flask import request

from configuration.Decorators import log_request_and_response
from controller.BaseController import BaseController
from services.TestFlowService import handle_request


class TestFlowController(BaseController):
    ENRICH_ROUTE = "/api/test"  # Define the common part of the route

    def setup_routes(self):
        @self.app.route(f"{self.ENRICH_ROUTE}/selector_model", methods=["GET"])
        @log_request_and_response
        def test_selector_model():
            handle_request(request)
