import logging

from controller.BaseController import BaseController


class HealthCheckController(BaseController):
    def setup_routes(self):
        @self.app.route("/api/healthcheck", methods=["GET"])
        def health_check():
            logging.info("Server is alive...")
            return 'hello, Im alive', 200
