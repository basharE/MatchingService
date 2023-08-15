import tensorflow as tf
from transformers import CLIPProcessor, CLIPModel
import time
from tqdm import tqdm
from numpy import linalg as LA
import numpy as np
from numpy import asarray
from keras.applications.resnet import preprocess_input
from keras import utils
import logging


def get_pretrained_model(name):
    if name == "resnet":
        # High-level tensorflow.keras API to easily use pretrained models
        model = tf.keras.applications.ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            classes=1000,
            pooling="max",
            classifier_activation="softmax",
        )
    else:
        print("Specified model not available")
    return model


def get_features_clip(processor_, device_, model_, tmpImg):
    # tmpImg = Image.open(image_path_)
    image1 = processor_(text=None, images=tmpImg, return_tensors="pt")[
        "pixel_values"
    ].to(device_)
    tmpImg.close()
    e = model_.get_image_features(image1)
    e = e.squeeze(0)
    e = e.cpu().detach().numpy()
    return e


def run_model(model_to_run, image_path):
    feature_list = []
    start_time = time.time()
    image1 = utils.load_img(image_path, target_size=(224, 224))
    if model_to_run == "resnet":
        transformedImage = asarray(image1)
        transformedImage = np.expand_dims(transformedImage, axis=0)
        transformedImage = preprocess_input(transformedImage)

        model = get_pretrained_model("resnet")

        # Model predictions are feature vectors as the final
        # classification layer is removed
        feature_list = model.predict(transformedImage)

    elif model_to_run == "clip":
        # if you have CUDA or MPS, set it to the active device like this
        # device = "cuda" if torch.cuda.is_available() else \
        #         ("mps" if torch.backends.mps.is_available() else "cpu")
        device = "cpu"
        model_id = "openai/clip-vit-base-patch32"

        # we initialize a tokenizer, image processor, and the model itself
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)

        e = get_features_clip(processor, device, model, image1)
        feature_list.append(e)

    end_time = time.time()
    # Features are normalized to avoid scaling issues.
    # It is usually a good practice to constrain the range of values
    # a feature might take to ensure one or a few dimensions do not
    # dominate the overall feature space.
    for i, features in tqdm(enumerate(feature_list)):
        feature_list[i] = features / LA.norm(features)

    # Convert list to numpy array
    feature_list = np.array(feature_list)
    logging.info("Type = %s", type(feature_list))
    logging.info("Shape of feature_list = %s", feature_list.shape)
    logging.info("Time taken in sec = %s", end_time - start_time)
    logging.info("Image location: %s", image_path)
    return feature_list
