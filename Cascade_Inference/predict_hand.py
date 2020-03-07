#-*-coding:utf-8-*-
# date:2020-03-02
# Author: X.li
# function: inference backbone only support resnet

import os
import glob
import cv2
import numpy as np
import time
import shutil
import torch
import json
import sys
sys.path.append('./Hand/')
from Hand.hand_detect import Hand_Run

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]# color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # filled rectangle
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],\
        thickness=tf, lineType=cv2.LINE_AA)
if __name__ == '__main__':
	model_arch = 'resnet_18'
	model_path = '../Hand_Detect/model_save/model_hand_last_'+model_arch+'.pth'# 模型路径
	img_dir = '../done/'# 测试集
	nms_thr = 0.75
	nms_flag = True

	model_hand_ = Hand_Run(model_arch,nms_flag,nms_thr,model_path)
	colors = [(55,55,250), (255,155,50), (128,0,0), (255,0,255), (128,255,128), (255,0,0)]

	for file_ in os.listdir(img_dir):
		if '.xml' in file_:
			continue
		print("--------------------")
		img = cv2.imread(img_dir + file_)
		hm,detections_hands = model_hand_.predict(img)
		print('model_arch - {} : {}'.format(model_arch,detections_hands))
		if len(detections_hands) > 0 :
			for i in range(len(detections_hands)):
				bbox_,cls_,conf_ = detections_hands[i]

				if nms_flag:
					label_ = 'nms_{}:{:.2f}'.format(cls_,conf_)
					plot_one_box(bbox_, img, color=(55,125,255), label=label_, line_thickness=2)
				else:
					label_ = '{}:{:.2f}'.format(cls_, conf_)
					plot_one_box(bbox_, img, color=(55,125,255), label=label_, line_thickness=2)
		cv2.namedWindow("heatmap", 0)
		cv2.imshow("heatmap", np.hstack(hm[0].cpu().numpy()))
		cv2.namedWindow("img", 0)
		cv2.imshow("img", img)
		key = cv2.waitKey(1)
		if key == 27:
			break
