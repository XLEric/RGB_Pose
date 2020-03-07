#coding:utf-8
# date:2019-08
# Author: X.li
# function: predict

import argparse
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import cv2
import numpy as np
from Person.person_detect import Person_Run

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':

    model_path = "../Person_Detect/weights-yolov3/latest.pt" # 检测模型路径
    root_path = '../Person_Detect/example_person/'# 测试文件夹
    voc_config = './Person/cfg_person/voc_person.data' # 模型相关配置文件
    img_size = 416 # 图像尺寸
    conf_thres = 0.3# 检测置信度
    nms_thres = 0.4 # nms 阈值
    model_cfg = 'YOLOv3'

    # colors = [(0,0,255),(0,255,0)]
    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, 20 + 1)][::-1]
    model_person_ = Person_Run(\
    data_cfg = voc_config,\
    model_cfg = model_cfg,\
    model_path= model_path,\
    img_size = img_size,\
    conf_thres = 0.35,\
    nms_thres = 0.3,\
    )

    idx = 0
    for img_name in os.listdir(root_path):
        idx += 1
        img_path  = root_path + img_name
        im_ = cv2.imread(img_path)
        im0 = im_.copy()
        detections = model_person_.predict(im0)

        if detections is not None:
            print('------>>>  ',idx)
            for *xyxy, conf, cls_conf, cls in detections:
                label = model_person_.classes[int(cls)]
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                x1,y1,x2,y2 = xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()
                print('x1,y1,x2,y2:',x1,y1,x2,y2)
                x1 = int(max(x1,0))
                y1 = int(max(y1,0))
                x2 = int(min(x2,im0.shape[1]-1))
                y2 = int(min(y2,im0.shape[0]-1))
                #----------------------------------
                bbox_w = int((x2-x1)*0.08)
                bbox_h = int((y2-y1)*0.05)

                x1 = max((0,x1-bbox_w))
                y1 = max((0,y1-bbox_h))
                x2 = min((im0.shape[1]-1,x2+bbox_w))
                y2 = min((im0.shape[0]-1,y2+bbox_h))

                img_person_crop = im_[y1:y2,x1:x2,:]

                cv2.namedWindow('result',0)
                cv2.imshow("result", im0)
                cv2.namedWindow('person',0)
                cv2.imshow("person", img_person_crop)
                cv2.waitKey(0)

                print('   label: %s , score : %.3f) '%(label,conf.item()))
        else:
            print(" --------------------- no detect something",im0.shape)

        cv2.namedWindow('result',0)
        cv2.imshow("result", im0)
        key = cv2.waitKey(0)
        if key == 27:
            break
