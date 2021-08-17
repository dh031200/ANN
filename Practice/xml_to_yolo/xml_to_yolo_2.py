import xml.etree.ElementTree as ET
import os
from labels import label_to_idx


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    R = [x, y, w, h]
    return ' '.join(map(str, R))


root_dir = 'Bbox_1_new/'
list_dir = os.listdir(root_dir)

for i in list_dir:
    files = os.listdir(root_dir + i)
    xml_name = ''
    for j in files:
        if j[-3:] == 'xml':
            xml_name = j
            break
    doc = ET.parse(root_dir + i + '/' + xml_name)

    images = doc.getroot().findall('image')
    image_list = []
    for image in images:
        name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        f = open(root_dir + i + '/' + name[:-3] + 'txt', 'w')
        for box in image.findall('box'):
            label = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            annots = convert((width, height), [xtl, ytl, xbr, ybr])
            f.write(label_to_idx(label) + ' ' + annots + '\n')
        f.close()