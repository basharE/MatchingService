import cv2
import os

from configuration.ConfigurationService import get_frames_directory_from_conf
from utils.PathUtils import create_path
import logging

from utils.Common import remove_extension


def extract_frame(vid_cap, sec, count, video_name):
    vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    has_frames, image = vid_cap.read()
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if has_frames:
        filename = os.path.join(get_frames_directory_from_conf(), video_name, f"image{count}.jpg")
        cv2.imwrite(filename, image)  # save frame as JPG file

        # Display the extracted frame
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title(f"Frame {count}")
        # plt.show()
    return has_frames


def split_images_stream(images_stream, video_name):
    logging.info('Starting split_images_stream for file: %s', video_name)

    video_name_ = remove_extension(video_name)
    create_path(os.path.join(get_frames_directory_from_conf(), video_name_))

    frame_rate = 0.5  # it will capture an image every 0.5 seconds
    count = 1
    sec = 0

    while extract_frame(images_stream, sec, count, video_name_):
        count += 1
        sec += frame_rate
        sec = round(sec, 2)

    logging.info('Completed processing split_images_stream for file: %s', video_name)
