label_list = [
    'bicycle',
    'bus',
    'car',
    'carrier',
    'cat',
    'dog',
    'motorcycle',
    'movable_signage',
    'person',
    'scooter',
    'stroller',
    'truck',
    'wheelchair',
    'barricade',
    'bench',
    'bollard',
    'chair',
    'fire_hydrant',
    'kiosk',
    'parking_meter',
    'pole',
    'potted_plant',
    'power_controller',
    'stop',
    'table',
    'traffic_light',
    'traffic_light_controller',
    'traffic_sign',
    'tree_trunk'
]


def label_to_idx(name):
    return str(label_list.index(name))
