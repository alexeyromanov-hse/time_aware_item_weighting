import os
import json
import itertools


def get_attribute(name, default_value=None):
    """
    get configs
    :param name:
    :param default_value:
    :return:
    """
    if "data_path" not in CONFIG:
        CONFIG['data_path'] = \
            f"{os.path.dirname(os.path.dirname(__file__))}/data/{CONFIG['data']}/{CONFIG['data']}.json"
        CONFIG['items_total'] = get_items_total(CONFIG['data_path'])
    try:
        return CONFIG[name]
    except KeyError:
        return default_value


def get_items_total(data_path):
    items_set = set()
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        for key in ["train", "validate", "test"]:
            data = data_dict[key]
            for user in data:
                items_set = items_set.union(set(itertools.chain.from_iterable(data[user])))
    return len(items_set)


CONFIG = {}
