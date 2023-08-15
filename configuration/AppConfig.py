from configparser import ConfigParser


class AppConfig:
    def __init__(self, config_file_path='app.config'):
        self.config_parser = ConfigParser()
        self.config_parser.read(config_file_path)
        self.app_config = self._load_config()

    def _load_config(self):
        app_configs = {}
        for section_name in self.config_parser.sections():
            app_configs[section_name] = {}
            for key, value in self.config_parser.items(section_name):
                app_configs[section_name][key] = value
        return app_configs

    def get_config(self):
        return self.app_config


# Usage
if __name__ == '__main__':
    config = AppConfig()
    app_config = config.get_config()
    print(app_config)
