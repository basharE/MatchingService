from configuration.AppConfig import AppConfig


def get_configuration_value(key_hierarchy):
    conf = AppConfig('configuration/app.config')
    config = conf.get_config()

    keys = key_hierarchy.split('.')
    value = config
    for key in keys:
        value = value.get(key, {})

    return value


def get_database_uri_from_conf():
    return get_configuration_value('database.uri')


def get_database_name_from_conf():
    return get_configuration_value('database.database_name')


def get_database_images_collection_name_from_conf():
    return get_configuration_value('database.images_collection_name')


def get_database_train_collection_name_from_conf():
    return get_configuration_value('database.train_collection_name')


def get_database_new_train_collection_name_from_conf():
    return get_configuration_value('database.new_train_collection_name')


def get_frames_directory_from_conf():
    return get_configuration_value('video.frames_directory')


def get_database_collection_name_from_conf():
    return get_configuration_value('database.images_collection_name')


def get_directory_from_conf():
    return get_configuration_value('video.video_directory')


def get_clip_threshold_const_from_conf():
    return get_configuration_value('video.clip_threshold_const')


def get_resnet_threshold_const_from_conf():
    return get_configuration_value('video.resnet_threshold_const')


def get_image_directory_from_conf():
    return get_configuration_value('video.image_directory')


def get_clip_threshold_from_conf():
    return float(get_configuration_value('training.clip_threshold'))


def get_resnet_threshold_from_conf():
    return float(get_configuration_value('training.resnet_threshold'))


def get_number_of_highest_results_from_conf():
    return int(get_configuration_value('training.number_of_highest_results'))
