from flask import Flask


class BaseController:
    def __init__(self, app, config):
        self.app = app
        self.config = config
        self.setup_routes()

    def setup_routes(self):
        raise NotImplementedError("Subclasses must implement setup_routes")
