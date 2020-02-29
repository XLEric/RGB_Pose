#-*-coding:utf-8-*-
# date:2019-12-10
# Author: xiang li
# function: read hand labels

import os
import json
import cv2
import numpy as np

# path_ = 'H:/Open_Datasets/cmp_hand/hand_labels/manual_train/'
path_ = 'H:/Open_Datasets/cmp_hand/hand_labels_synth/synth2/'

colors = [[0, 0, 255],[0, 0, 255],[0, 0, 255],[0, 0, 255],
    [0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0],
    [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],
    [0, 215, 215],[0, 215, 215],[0, 215, 215],[0, 215, 215],
    [255, 0, 255],[255, 0, 255],[255, 0, 255],[255, 0, 255]]

linkSeq = [[0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
    ]
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
if __name__ == "__main__":

    idx = 0
    font = cv2.FONT_HERSHEY_PLAIN

    num_classes = 21

    for i,file_ in enumerate(os.listdir(path_)):
        if '.json' in file_:
            file_path = path_ + file_
            img_path = file_path.replace('.json','.jpg')
            if not os.path.exists(img_path):
                continue

            img_ = cv2.imread(img_path)

            f = open(file_path, encoding='utf-8')#读取 json文件
            object_dict_ = json.load(f)
            f.close()

            # print(object_dict_)

            for key_ in object_dict_.keys():
                # print(key_)
                if key_ == 'hand_pts':
                    print(len(object_dict_['hand_pts']))
                    x_min = 65535
                    y_min = 65535
                    x_max = 0
                    y_max = 0
                    for pts_ in object_dict_['hand_pts']:
                        x_,y_,see_ = int(pts_[0]),int(pts_[1]),pts_
                        if see_:
                            x_min = int(x_ if x_<x_min else x_min)
                            y_min = int(y_ if y_<y_min else y_min)
                            x_max = int(x_ if x_>x_max else x_max)
                            y_max = int(y_ if y_>y_max else y_max)
                            cv2.circle(img_, (x_,y_), 3, (255,50,100), -1)

                    bbox_w = int((x_max-x_min)*0.15)
                    bbox_h = int((y_max-y_min)*0.15)
                    print('1) ',x_min,y_min,x_max,y_max)
                    x_min = max((0,x_min-bbox_w))
                    y_min = max((0,y_min-bbox_h))
                    x_max = min((img_.shape[1]-1,x_max+bbox_w))
                    y_max = min((img_.shape[0]-1,y_max+bbox_h))
                    print('2) ',x_min,y_min,x_max,y_max)

                    bbox_ = (x_min,y_min,x_max,y_max)
                    plot_one_box(bbox_,img_, label='Hand', color=(255,211,11))

                    for  k in range(len(linkSeq)):
                        link = linkSeq[k]
                        [X0,Y0,see_0] = object_dict_['hand_pts'][link[0]]
                        [X1,Y1,see_1] = object_dict_['hand_pts'][link[1]]
                        if see_0 and see_1:
                            cv2.line(img_, (int(X0),int(Y0)), (int(X1),int(Y1)), colors[k], 2)


            cv2.namedWindow('image',0)
            cv2.imshow('image',img_)
            key_id = cv2.waitKey(0)
            if key_id == 27:
                break

    cv2.destroyAllWindows()
