import logging

from configuration.Decorators import log_request_and_response
from controller.BaseController import BaseController
from flask import make_response


class HealthCheckController(BaseController):
    def setup_routes(self):
        @self.app.route("/api/healthcheck", methods=["GET"])
        @log_request_and_response
        def health_check():
            logging.info("Server is alive...")
            response_data = 'hello, Im alive'
            status_code = 200
            response = make_response(response_data, status_code)
            return response
