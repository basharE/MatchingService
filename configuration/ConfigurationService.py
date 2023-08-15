from configuration.AppConfig import AppConfig


def get_configuration():
    conf = AppConfig('configuration/app.config')
    return conf.get_config()


def get_database_uri_from_conf():
    return get_configuration().get('database').get('uri')


def get_database_name_from_conf():
    return get_configuration().get('database').get('database_name')


def get_database_images_collection_name_from_conf():
    return get_configuration().get('database').get('images_collection_name')


def get_database_train_collection_name_from_conf():
    return get_configuration().get('database').get('train_collection_name')


def get_video_conf():
    conf = AppConfig('configuration/app.config')
    return conf.get_config().get('video')


def get_frames_directory_from_conf():
    return get_video_conf().get('frames_directory')


def get_database_collection_name_from_conf():
    return get_configuration().get('database').get('images_collection_name')


def get_directory_from_conf():
    return get_video_conf().get('video_directory')


def get_threshold_const_from_conf():
    return get_video_conf().get('threshold_const')
