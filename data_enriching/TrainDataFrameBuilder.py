from collections import ChainMap

from data_enriching.ZoneService import Zone


def get_best_k_of_types(k, types, item):
    similarities_of_all_type = {}
    for the_type in types:
        similarities_of_type = {}
        for similarity in item.items():
            similarity_key, similarity_value = similarity
            if isinstance(similarity_value, float) and the_type in similarity_key.lower():
                similarities_of_type[similarity_key] = similarity_value
        sorted_dict_by_value = {key: value for key, value in
                                sorted(similarities_of_type.items(), key=lambda ite: ite[1])}
        first_k_items = dict(list(sorted_dict_by_value.items())[0: k])
        similarities_of_all_type.update(first_k_items)
    return similarities_of_all_type


def get_percent_of_types_for_threshold(thresholds, types, item):
    percent_bellow_threshold = {}
    for the_type in types:
        bellow_threshold_counter = 0
        for similarity in item.items():
            similarity_key, similarity_value = similarity
            if isinstance(similarity_value,
                          float) and the_type in similarity_key.lower() and similarity_value < thresholds[the_type]:
                bellow_threshold_counter = bellow_threshold_counter + 1
        percent_bellow_threshold[f"{the_type}_bellow_threshold"] = bellow_threshold_counter
    return percent_bellow_threshold


def get_zone(zone_name):
    zone_ = Zone()
    zone_represented_number = zone_.get_zone_id(zone_name)
    return {'zone': zone_represented_number}


# need to build a request object that will contain initial metadata related to request (zone, name, description ...)
# then it will be enriched with data like number of frames, video length ...
def find_best_k_results(similarities):
    threshold = {'clip': 0.55, 'resnet': 0.25}
    k = 3
    types = ["clip", "resnet"]
    similarities_as_data_frame = {}
    for item in similarities.items():
        item_key, item_value = item
        best_k_results = get_best_k_of_types(k, types, item_value)
        percent_of_types = get_percent_of_types_for_threshold(threshold, types, item_value)
        best_k_results.update(percent_of_types)
        concatenated_dict = dict(ChainMap(best_k_results, percent_of_types, get_zone(item_value['zone'])))
        similarities_as_data_frame[item_key] = concatenated_dict

    return similarities_as_data_frame
