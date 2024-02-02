import tensorflow as tf
import torch
from transformers import CLIPProcessor, CLIPModel
import time
from tqdm import tqdm
from numpy import linalg as LA
import numpy as np
from numpy import asarray
from keras.applications.resnet import preprocess_input
from keras import utils
import logging

from data_enriching.FeaturesServices import initialize_orb_detector, extract_descriptors_


def get_features_clip(processor_, device_, model_, tmp_img):
    image1 = processor_(text=None, images=tmp_img, return_tensors="pt")[
        "pixel_values"].to(device_)
    with torch.no_grad():
        features = model_.get_image_features(image1).squeeze(0).cpu().numpy()

    return features


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FeatureExtractor(metaclass=SingletonMeta):

    def __init__(self):
        self.processor = None
        # self.resnet_model = self.get_pretrained_model("resnet")
        self.clip_model = self.get_pretrained_model("clip")
        # self.orb_model = self.get_pretrained_model("orb")

    def get_pretrained_model(self, name):
        if name == "resnet":
            # High-level tensorflow.keras API to easily use pretrained models
            model = tf.keras.applications.ResNet152V2(
                weights='imagenet',
                include_top=False,
                # input_shape=(224, 224, 3),
                classes=1000,
                pooling="max",
                classifier_activation="softmax",
            )
        elif name == "clip":
            device = "cpu"
            # model_id = "openai/clip-vit-large-patch14"
            model_id = "openai/clip-vit-large-patch14"
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id).to(device)
            self.processor = processor

        elif name == "orb":
            # Initialize feature detector
            model = initialize_orb_detector()
        else:
            print("Specified model not available")
        return model

    def run_resnet_model(self, image_path):
        model = self.resnet_model
        image1 = utils.load_img(image_path)

        transformed_image = asarray(image1)
        transformed_image = np.expand_dims(transformed_image, axis=0)
        transformed_image = preprocess_input(transformed_image)

        feature_list = model.predict(transformed_image)

        return feature_list

    def run_clip_model(self, image_path):
        device = "cpu"
        feature_list = []
        image1 = utils.load_img(image_path)
        e = get_features_clip(self.processor, device, self.clip_model, image1)
        feature_list.append(e)

        return np.array(feature_list)

    def run_orb_model(self, image_path):
        return extract_descriptors_(image_path, self.orb_model)

    def run_model_(self, model_to_run, image_path):
        if model_to_run == "resnet":
            return self.run_resnet_model(image_path)
        elif model_to_run == "clip":
            return self.run_clip_model(image_path)
        elif model_to_run == "orb":
            return self.run_orb_model(image_path)
        else:
            print("Specified model not available")
            return []

    def run_model(self, model_to_run, image_path):
        start_time = time.time()
        feature_list = self.run_model_(model_to_run, image_path)
        end_time = time.time()

        # Features are normalized to avoid scaling issues.
        # It is usually a good practice to constrain the range of values
        # a feature might take to ensure one or a few dimensions do not
        # dominate the overall feature space.
        if model_to_run != "orb":
            for i, features in tqdm(enumerate(feature_list)):
                feature_list[i] = features / LA.norm(features)

            # Convert list to numpy array
            feature_list = np.array(feature_list)

        logging.info("Type = %s", type(feature_list))
        logging.info("Shape of feature_list = %s", feature_list.shape)
        logging.info("Time taken in sec = %s", end_time - start_time)
        logging.info("Image location: %s", image_path)
        return feature_list
