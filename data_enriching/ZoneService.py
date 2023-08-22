class Zone:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Zone, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.zone_dictionary = {}

    def get_zone_id(self, zone_name):
        if self.zone_dictionary.get(zone_name) is not None:
            return self.zone_dictionary[zone_name]
        else:
            return self.create_entry(zone_name)

    def create_entry(self, zone_name):
        _value = self.get_max_value()
        new_value = _value + 1
        self.zone_dictionary[zone_name] = new_value
        return new_value

    def get_max_value(self):
        _max = 0
        for value in self.zone_dictionary.values():
            if value > _max:
                _max = value
        return _max
