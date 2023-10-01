from configuration.ConfigurationService import get_image_directory_from_conf
from image_find.FrameHandler import extract_features, find_similarities
from utils.PlotUtil import plot_line_graph


def get_for_graph(images_similarities):
    clip_values = list()
    resnet_values = list()
    object_indicator_for_clip = list()
    object_indicator_for_resnet = list()
    for k, v in images_similarities.items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                if "clip" in _k:
                    clip_values.append(_v)
                    object_indicator_for_clip.append(k)
                if "resnet" in _k:
                    resnet_values.append(_v)
                    object_indicator_for_resnet.append(k)
    return [i for i in range(len(clip_values))], clip_values, object_indicator_for_clip, [i for i in range(
        len(resnet_values))], resnet_values, object_indicator_for_resnet


def handle_request(request):
    image = request.files['image']
    image_features = extract_features(image, get_image_directory_from_conf())
    images_similarities = find_similarities(image_features, 0)

    clip_index_list, clip_similarity_list, class_clip, resnet_index_list, resnet_similarity_list, class_resnet = get_for_graph(
        images_similarities)

    # Plotting
    plot_line_graph(clip_index_list, clip_similarity_list, resnet_index_list, resnet_similarity_list, class_clip,
                    "uploads/plot/" + image.filename + "-graph.png", image.filename)
    return ""
