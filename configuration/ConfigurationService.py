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


def get_frames_directory_from_conf():
    return get_configuration_value('video.frames_directory')


def get_database_collection_name_from_conf():
    return get_configuration_value('database.images_collection_name')


def get_directory_from_conf():
    return get_configuration_value('video.video_directory')


def get_threshold_const_from_conf():
    return get_configuration_value('video.threshold_const')


def get_image_directory_from_conf():
    return get_configuration_value('video.image_directory')
