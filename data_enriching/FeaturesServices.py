import collections
import cv2
from datetime import datetime
from scipy.spatial.distance import euclidean

root_dir = 'C:/Users/Bashar/Documents/Thesis/Thesis/classifier_data/data/latest' + str(1) + '/'


# method that find the image place in the given vedio
# by finding the most close frame to each image in the
# validation collection

# [0]validation_list_indexes,
# [1]frames_list_converted,
# [2]scores_list,
# [3]filtered_frames_list_converted,
# [4]scores_list_with_filtered_frames,
# [5]frames_list


def guess_final_place(index, feature_list_resnet, feature_list_clip, generator):
    result = list()
    # calculate distances using both resnet and clip models
    result_resnet = calculate_distance_of_features(index, feature_list_resnet, generator)
    result_clip = calculate_distance_of_features(index, feature_list_clip, generator)

    # print("processing data for file"+generator.filenames[result_resnet[0][index]])
    # print("processing data for file"+generator.filenames[result_clip[0][index]])
    # sort resnet results
    key_value_dict_resnet = dict(zip(result_resnet[5], result_resnet[2]))
    sorted_key_value_dict_resnet = dict(
        sorted(key_value_dict_resnet.items(), key=lambda item: item[1])
    )
    sorted_dict_resnet = collections.OrderedDict(sorted_key_value_dict_resnet)
    # sort clip results
    key_value_dict_clip = dict(zip(result_clip[5], result_clip[2]))
    sorted_key_value_dict_clip = dict(
        sorted(key_value_dict_clip.items(), key=lambda item: item[1])
    )
    sorted_dict_clip = collections.OrderedDict(sorted_key_value_dict_clip)

    # get minimum from resnet and clip results
    resnet_frame_time_ = list(sorted_dict_resnet.keys())[0]
    clip_frame_time_ = list(sorted_dict_clip.keys())[0]

    resnet_frame_time = datetime.strptime(
        resnet_frame_time_.split("frame_", 2)[1].replace(".jpg", ""),
        "%d-%b-%Y (%H_%M_%S.%f)",
    )
    clip_frame_time = datetime.strptime(
        clip_frame_time_.split("frame_", 2)[1].replace(".jpg", ""),
        "%d-%b-%Y (%H_%M_%S.%f)",
    )

    # difference between suggested framed from clip and resnet in seconds
    diff_frames_times = abs((resnet_frame_time - clip_frame_time).total_seconds())

    # print(resnet_frame_time)
    # print(clip_frame_time)

    # print(diff_frames_times)

    resnet_best_accuracy = sorted_dict_resnet.get(list(sorted_dict_resnet.keys())[0])
    clip_best_accuracy = sorted_dict_clip.get(list(sorted_dict_clip.keys())[0])

    # print(resnet_best_accuracy)
    # print(clip_best_accuracy)

    result.append(generator.filenames[result_resnet[0][index]])
    result.append(resnet_frame_time_)
    result.append(resnet_best_accuracy)
    avg = -1
    # if clip_best_accuracy>0.4 and clip_best_accuracy<0.55: filtered_frames_list=[list(sorted_dict_clip.keys())[0],
    # list(sorted_dict_clip.keys())[1],list(sorted_dict_clip.keys())[2],list(sorted_dict_clip.keys())[3] ,
    # list(sorted_dict_clip.keys())[4],list(sorted_dict_clip.keys())[5],list(sorted_dict_clip.keys())[6],
    # list(sorted_dict_clip.keys())[7], list(sorted_dict_clip.keys())[8],list(sorted_dict_clip.keys())[9]]
    filtered_frames_list_resnet = [list(sorted_dict_resnet.keys())[0]]
    scores_list_with_filtered_frames_resnet = get_indicator_to_closest_frame(
        filtered_frames_list_resnet, generator.filenames[result_resnet[0][index]]
    )
    # scores_list_with_filtered_frames = [-1,-1]
    avg_resnet = sum(scores_list_with_filtered_frames_resnet) / len(
        scores_list_with_filtered_frames_resnet
    )

    result.append(avg_resnet)
    result.append(clip_frame_time_)
    result.append(clip_best_accuracy)

    filtered_frames_list = [list(sorted_dict_clip.keys())[0]]
    scores_list_with_filtered_frames = get_indicator_to_closest_frame(filtered_frames_list,
                                                                      generator.filenames[result_resnet[0][index]])
    # scores_list_with_filtered_frames = [-1,-1]
    avg = sum(scores_list_with_filtered_frames) / len(scores_list_with_filtered_frames)

    result.append(scores_list_with_filtered_frames)
    result.append(avg)
    #   else:
    #     scores_list_with_filtered_frames = [-1,-1]
    #     result.append(scores_list_with_filtered_frames[0])
    #     result.append(clip_frame_time_)
    #     result.append(clip_best_accuracy)
    #     result.append(scores_list_with_filtered_frames[1])
    #     result.append(0)
    if clip_best_accuracy < 0.4:
        result.append("image located!")
        result.append(clip_frame_time_)
    elif (diff_frames_times < 2
          and resnet_best_accuracy < 0.8
          and clip_best_accuracy < 0.5
          and 0.8 > avg > 0):
        result.append("image located!")
        result.append(clip_frame_time_)
    else:
        result.append("frame cannot be located")
        result.append("empty")

    result.append(diff_frames_times)

    return result


def calculate_distance_of_features(index, feature_list, generator):
    validation_list_indexes = list()
    for i in range(len(generator.filenames)):
        item = generator.filenames[i]
        if "validation" in item:
            validation_list_indexes.append(i)

    frames_list = list()
    scores_list = list()
    result = list()
    result.append(["frame_name", "euclidean_dis"])

    validation_features = feature_list[validation_list_indexes[index]]
    # print(validation_features)
    for i in range(len(generator.filenames)):
        frame_item = generator.filenames[i]
        if "frames" in frame_item:
            frame_features = feature_list[i]
            euclidean_dis = euclidean(validation_features, frame_features)
            result.append([frame_item, euclidean_dis])
            frames_list.append(frame_item)
            scores_list.append(euclidean_dis)
    ##################################################################
    frames_list_converted = frame_txt_date(frames_list)
    # print(scores_list)

    filtered_frames_list = filter_bad_frames(frames_list, scores_list)
    scores_list_with_filtered_frames = []
    # scores_list_with_filtered_frames = get_indicator_to_closest_frame(filtered_frames_list, generator.filenames[
    # validation_list_indexes[index]])

    filtered_frames_list_converted = frame_txt_date(filtered_frames_list)
    # print("****************************** ")
    # print(frames_list_converted)
    # print("****************************** ")
    # print(scores_list)
    # print("****************************** ")
    # print(filtered_frames_list_converted)
    # print("****************************** ")
    # print(scores_list_with_filtered_frames)

    result = [
        validation_list_indexes,
        frames_list_converted,
        scores_list,
        filtered_frames_list_converted,
        scores_list_with_filtered_frames,
        frames_list,
    ]

    return result


def get_indicator_to_closest_frame(filtered_frames_list_, validation_image_):
    # calculate similarities for all frames using opencv
    opencv_scores = list()
    for i in range(len(filtered_frames_list_)):
        concated_validation_image_ = root_dir + validation_image_
        concated_filtered_frames_list_ = root_dir + filtered_frames_list_[i]
        opencv_scores.append(
            calculate_similarity(
                concated_validation_image_, concated_filtered_frames_list_, 2
            )
        )
    if len(opencv_scores) == len(filtered_frames_list_):
        # print("finish with size of ", len(opencv_scores))
        return opencv_scores


def calculate_similarity(img1_, img2_, algo):
    # upload images
    img1 = cv2.imread(img1_, 0)
    img2 = cv2.imread(img2_, 0)

    # Initiate SIFT detector
    if algo == 0:
        algorithem = cv2.xfeatures2d.SIFT_create()
    if algo == 1:
        algorithem = cv2.xfeatures2d.SURF_create()
    if algo == 2:
        algorithem = cv2.ORB_create(nfeatures=100)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = algorithem.detectAndCompute(img1, None)
    kp2, des2 = algorithem.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append([m])
    score = 1 - (len(good) / len(matches))

    return score


def frame_txt_date(input_frames_list):
    sep1 = "frame_"
    sep2 = "frame"
    frames_list_converted = list()
    for i in range(len(input_frames_list)):
        date_time_obj = ""
        if sep1 in input_frames_list[i]:
            date_time_obj = datetime.strptime(
                input_frames_list[i].split(sep1, 2)[1].replace(".jpg", ""),
                "%d-%b-%Y (%H_%M_%S.%f)",
            )
        else:
            date_time_obj = datetime.strptime(
                input_frames_list[i].split(sep2, 2)[2].replace(".jpg", ""),
                "%d-%b-%Y (%H_%M_%S.%f)",
            )
        frames_list_converted.append(date_time_obj)
    return frames_list_converted


def filter_bad_frames(frames_list_, scores_list_):
    # print("-----------------------------------")
    key_value_dict = dict(zip(frames_list_, scores_list_))

    sorted_key_value_dict = dict(
        sorted(key_value_dict.items(), key=lambda item: item[1])
    )

    sorted_dict = collections.OrderedDict(sorted_key_value_dict)

    # print(sorted_dict)

    filtered_frames_list = list()
    if len(frames_list_) == len(scores_list_):
        count = 0
        PERCENTAGE_TOP_X_FRAMES = 3 / 1000
        TOP_X_FRAMES = PERCENTAGE_TOP_X_FRAMES * len(scores_list_)
        # print(len(scores_list_))
        # print(TOP_X_FRAMES)
        while count < TOP_X_FRAMES:
            filtered_frames_list.append(list(sorted_dict)[count])
            count = count + 1

    # print("-----------------------------------")
    return filtered_frames_list
