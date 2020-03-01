#-*-coding:utf-8-*-
# date:2019-06
# Author: X.li
# function: make train datasets

import os
import os.path
import xml.etree.cElementTree as et
import cv2
import numpy as np
import shutil

if __name__ == "__main__":
    path_data = 'G:/project/yolact/detect_person_datasets/'
    path_datasets_  = 'datasets_person/'
    if not os.path.exists(path_datasets_):
        os.mkdir(path_datasets_)

    choose_list = ['person']

    label_dict_ = {}
    label_cnt_dict_ = {}

    for i in range(len(choose_list)):
        label_dict_[choose_list[i]] = i
        label_cnt_dict_[choose_list[i]] = 0

    print('label_dict_',label_dict_)

    if not os.path.exists(path_datasets_+'anno/'):
        os.mkdir(path_datasets_ + 'anno/')

    save_images_path = path_datasets_ + 'anno/images/'
    save_labels_path = path_datasets_ + 'anno/labels/'

    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    if not os.path.exists(save_labels_path):
        os.mkdir(save_labels_path)

    idx = 0

    train_ = open(path_datasets_ + 'anno/train.txt', 'w')


    for file in os.listdir(path_data):

        if os.path.splitext(file)[1]==".jpg" or os.path.splitext(file)[1]==".png":
            # if idx >= 2000:
            #     break
            image_path_o =  path_data + file
            xml_path_o = image_path_o.replace('.jpg','.xml').replace('.png','.xml')
            image_path = save_images_path + file

            txt_path = image_path.replace('images','labels').replace('.jpg', '.txt').replace('.png', '.txt')
            if not os.path.exists(xml_path_o):
                continue

            shutil.copy(image_path_o,image_path)

            idx += 1
            flag_txt = False
            #---------------------------------------------------------
            tree=et.parse(xml_path_o)
            root=tree.getroot()

            img = cv2.imread(image_path)
            try:
                height = img.shape[0]
                width =img.shape[1]
                print(idx,') exist --->>>',xml_path_o)
            except:
                print('-------->>> image error ')
                continue

            for Object in root.findall('object'):
                name=Object.find('name').text

                bndbox=Object.find('bndbox')
                xmin= np.float32((bndbox.find('xmin').text))
                ymin= np.float32((bndbox.find('ymin').text))
                xmax= np.float32((bndbox.find('xmax').text))
                ymax= np.float32((bndbox.find('ymax').text))
                # 归一化坐标
                x_mid = (xmax + xmin)/2./float(width)
                y_mid = (ymax + ymin)/2./float(height)

                w_box = (xmax-xmin)/float(width)
                h_box = (ymax-ymin)/float(height)

                label_xx = label_dict_[name]
                label_cnt_dict_[name] += 1

                if name in choose_list:
                    if flag_txt == False:
                        flag_txt = True
                        anno_txt = open(txt_path, 'w')
                    # print('-- label ',name)
                    anno_txt.write(str(label_xx)+' '+str(x_mid)+' '+str(y_mid)+' '+str(w_box)+' '+str(h_box)+ '\n')

            #---------------------------------------------------------
            if flag_txt == True:
                anno_txt.close()
                train_.write(image_path + '\n')
    train_.close()

    for key in label_cnt_dict_.keys():
        print('%s : %s'%(key,label_cnt_dict_[key]))
