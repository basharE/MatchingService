import cv2
import os

from configuration.AppConfig import AppConfig
from images_selector.PathUtils import create_path
import logging


def get_video_conf():
    conf = AppConfig('configuration/app.config')
    return conf.get_config().get('video')


def get_frames_directory_from_conf():
    return get_video_conf().get('frames_directory')


def extract_frame(vid_cap, sec, count, video_name):
    vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vid_cap.read()
    if hasFrames:
        filename = os.path.join(get_frames_directory_from_conf(), video_name, f"image{count}.jpg")
        cv2.imwrite(filename, image)  # save frame as JPG file
    return hasFrames


def split_images_stream(images_stream, video_name):
    logging.info('Starting split_images_stream for file: %s', video_name)

    video_name_ = remove_extension(video_name)
    create_path(os.path.join(get_frames_directory_from_conf(), video_name_))

    frameRate = 0.5  # it will capture an image every 0.5 seconds
    count = 1
    sec = 0

    while extract_frame(images_stream, sec, count, video_name_):
        count += 1
        sec += frameRate
        sec = round(sec, 2)

    logging.info('Completed processing split_images_stream for file: %s', video_name)


def remove_extension(filename):
    if filename.endswith(".mp4"):
        return filename[:-4]
    return filename
