import io

from PIL import Image


class ImageWrapper:
    def __init__(self, image_bytes, filename="image.jpg"):
        self.image_buffer = io.BytesIO(image_bytes)
        self.filename = filename

    def save(self, image_path):
        image = Image.open(self.image_buffer)
        image.save(image_path, format='JPEG')
