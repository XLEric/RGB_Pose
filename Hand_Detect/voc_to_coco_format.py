#-*-coding:utf-8-*-
# date:2020-03-02
# Author:
# function: voc to coco for eval

import tqdm
import os
import sys
import json
import shutil
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def get_img_path(path, extend=".jpg"):
    img_list = []
    for fpath, dirs, fs in os.walk(path):
        for f in fs:
            img_path = os.path.join(fpath, f)
            if os.path.dirname(img_path) == os.getcwd():
                continue
            if not os.path.isfile(img_path):
                continue
            file_name, file_extend = os.path.splitext(os.path.basename(img_path))
            if file_extend == extend:
                img_list.append(img_path)
    return img_list

def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

START_BOUNDING_BOX_ID = 1

def convert(xml_list, json_file, message="converting"):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(tqdm.tqdm(xml_list, message)):
        # print("Processing %s"%(line))
        xml_f = line
        if not os.path.exists(xml_f.replace('.xml','.jpg')):
            continue
            print(xml_f)
        tree = ET.parse(xml_f)
        root = tree.getroot()

        tmp_category = []
        for obj in get(root, 'object'):
            tmp_category.append(get_and_check(obj, 'name', 1).text)
        intersection = [i for i in tmp_category if i in classes]
        if only_care_pre_define_categories and len(intersection) == 0:
            continue

        filename = os.path.basename(xml_f)[:-4] + ".jpg"
        image_id = 20200000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert(xmax > xmin), "xmax <= xmin, {}".format(line)
            assert(ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=2)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("-->>> find {} categories: {} \n-->>> your pre_define_categories {}: {}".format(len(all_categories), set(list(all_categories.keys())), len(pre_define_categories), set(list(pre_define_categories.keys()))))
    if set(list(all_categories.keys())) == set(list(pre_define_categories.keys())):
        print("they are same")
    else:
        print("they are different")
    print("category: id --> {}".format(categories))
    print("available images number: {}".format(len(json_dict["images"])))
    print("save annotation to: {}".format(json_file))


if __name__ == '__main__':

    jpg_xml_path = "../done"

    save_json_train = './hand_detect_gt.json'

    classes = ['Hand']

    only_care_pre_define_categories = True

    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1

    xml_list = get_img_path(jpg_xml_path, ".xml")
    if len(xml_list) == 0:
        print("cannot find xml in {}".format(jpg_xml_path))
        exit()
    else:
        print("find {} xml in {}".format(len(xml_list), jpg_xml_path))
    xml_list = np.sort(xml_list)
    np.random.seed(100)
    np.random.shuffle(xml_list)

    print("voc to coco ...")
    convert(xml_list, save_json_train, "convert")
    print('jpg_xml_path : {} '.format(jpg_xml_path))
