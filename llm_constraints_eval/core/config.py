import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Process a set of configuration file directories.')
    parser.add_argument('--cfg_file', default='', help='The path to the directory containing configuration files.')
    args = parser.parse_args()
    return args

class NamedDict(dict):
    def __init__(self, *args, **kwargs):
        super(NamedDict, self).__init__(*args, **kwargs)
        # self.raw = self.copy()
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

def build_named_dict(data):
    result = NamedDict()
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = build_named_dict(value)
        else:
            result[key] = value
    return result

def get_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = build_named_dict(cfg)
    cfg['cfg_file'] = config_file
    return cfg