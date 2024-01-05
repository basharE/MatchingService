import io

from PIL import Image


class ImageWrapper:
    def __init__(self, image_bytes, filename="image.jpg"):
        self.image_buffer = image_bytes
        self.filename = filename

    def save(self, image_path):
        # image = Image.open(self.image_buffer)
        self.image_buffer.save(image_path, format='JPEG')
