#-*-coding:utf-8-*-
# date:2020-03-02
# Author: X.li
# function: CenterNet data iterator

import numpy as np
import cv2
import os
import math
import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

import xml.etree.cElementTree as ET
def get_xml_msg(path):
    list_x = []
    tree=ET.parse(path)# 解析 xml 文件
    root=tree.getroot()
    for Object in root.findall('object'):
        name=Object.find('name').text
        #----------------------------
        bndbox=Object.find('bndbox')
        xmin= np.float32((bndbox.find('xmin').text))
        ymin= np.float32((bndbox.find('ymin').text))
        xmax= np.float32((bndbox.find('xmax').text))
        ymax= np.float32((bndbox.find('ymax').text))
        bbox = int(xmin),int(ymin),int(xmax),int(ymax)
        xyxy = xmin,ymin,xmax,ymax
        list_x.append((name,xyxy))
    return list_x

class LoadImagesAndLabels(data.Dataset):
  #num_classes = 80
  class_name = ['__background__', "Hand"]
  _valid_ids = [1]
  dict2num = {'Hand':1}
  num_classes = len(_valid_ids)

  default_resolution = [512, 512]# 设定分辨率

  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  '''
  class_name = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
  _valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]'''

  def __init__(self, state, path_ = '../done/'):
    super(LoadImagesAndLabels, self).__init__()
    self.show = False

    img_list = []
    anno_list = []
    for file_ in os.listdir(path_):
        # if '.jpg' in file_:
        #     img_path_ = path_ + file_
        #     xml_path_ = path_ + file_.replace('.jpg','.xml')
        #     if not os.path.exists(xml_path_):
        #         os.remove(img_path_)
        #         os.remove(xml_path_)
        #         print('xxxxx')
        #         continue
        #     else:
        #         print('aaaa')
        #     continue

        if '.xml' in file_:
            xml_path_ = path_ + file_
            img_path_ = path_ + file_.replace('.xml','.jpg')

            # print(xml_path_)
            # print(img_path_)
            # if not os.path.exists(img_path_):
            #     # os.remove(img_path_)
            #     # os.remove(xml_path_)
            #     continue

            list_x = get_xml_msg(xml_path_)
            if len(list_x)>0:
                anno_list.append(list_x)
                img_list.append(img_path_)

    self.img_list =img_list
    self.anno_list =anno_list

    self.max_objs = 60

    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    if self.num_classes <= 5:
      self.voc_color = [(0, 255, 0), (0, 0, 255), (128, 0, 0), (255, 0, 255), (128, 255, 128), (255, 0, 0)]
    else:
      self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]

    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.state = state

    self.num_samples = len(self.img_list)
    self.not_rand_crop = False
    self.flip = 0.5
    self.no_color_aug = False
    self.scale = 0.4
    self.shift = 0.1
    self.down_ratio = 4
    self.dense_wh = False
    self.cat_spec_wh = False
    self.reg_offset = True
    self.mse_loss = False
    self.debug = 0
    self.input_h = 512
    self.input_w = 512

  def __len__(self):
    return self.num_samples

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
      i *= 2
    return border // i

  def __getitem__(self, index):
    img_path = self.img_list[index]
    anns = self.anno_list[index]
    num_objs = min(len(anns), self.max_objs)
    # print('anns:\n',anns)
    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

    s = max(img.shape[0], img.shape[1]) * 1.0
    input_h, input_w = self.input_h, self.input_w

    flipped = False

    if not self.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
    else:
        sf = self.scale
        cf = self.shift
        c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

    if np.random.random() < self.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] = width - c[0] - 1

    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    if self.show:
        input_img = inp.copy()

    inp = (inp.astype(np.float32) / 255.)

    if not self.no_color_aug:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.down_ratio
    output_w = input_w // self.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.mse_loss else draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
        ann = anns[k]
        label,bbox = ann
        bbox = np.array(bbox)
        cls_id = int(self.dict2num[label]-1)

        # print('bbox,cls_id : ',(bbox),(cls_id))
        if flipped:
            bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

        if self.show:
            cv2.putText(input_img, label, (int(bbox[0]*self.down_ratio), int(bbox[1]*self.down_ratio)), cv2.FONT_HERSHEY_COMPLEX, 1, self.voc_color[cls_id], 1)
            cv2.rectangle(input_img, (int(bbox[0]*self.down_ratio), int(bbox[1]*self.down_ratio)), (int(bbox[2]*self.down_ratio), int(bbox[3]*self.down_ratio)), self.voc_color[cls_id], 1)

        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]


        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            radius = radius
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]  # ind[k]: 0~128*128-1, object index in 128*128
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
            cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
            if self.dense_wh:
                draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

    if self.show:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", input_img)
        cv2.namedWindow("heatmap", 0)
        cv2.imshow("heatmap", np.hstack(hm))
        cv2.waitKey(500)

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.dense_wh:
        hm_a = hm.max(axis=0, keepdims=True)
        dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        del ret['wh']
    elif self.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.reg_offset:
      ret.update({'reg': reg})
    if self.debug > 0 or not self.state == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
