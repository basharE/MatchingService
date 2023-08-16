from controller.BaseController import BaseController
from deciding_model import Model_Trainer


class ModelTrainingController(BaseController):
    def setup_routes(self):
        @self.app.route("/api/data/train", methods=["GET"])
        def train_model():
            Model_Trainer.train_model()
            return 'done', 200
