#-*-coding:utf-8-*-
# date:2020-03-02
# Author: X.li
# function: predict CenterNet only support resnet backbone

import os
import glob
import cv2
import numpy as np
import time
import shutil
import torch

from data_iterator import LoadImagesAndLabels
from models.decode import ctdet_decode
from utils.model_utils import load_model
from utils.post_process import ctdet_post_process
from msra_resnet import get_pose_net as resnet

def letterbox(img, height=512, color=(31, 31, 31)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img

class CtdetDetector(object):
    def __init__(self,model_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.num_classes = LoadImagesAndLabels.num_classes
        print('Creating model...')
        model_arch = 'resnet_34'
        if "resnet_" in model_arch:
            num_layer = int(model_arch.split("_")[1])
            self.model = resnet(num_layers=num_layer, heads={'hm': self.num_classes, 'wh': 2, 'reg': 2}, head_conv=64, pretrained=True)  # res_18
        else:
            print("model_arch error:", model_arch)

        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = np.array([[[0.40789655, 0.44719303, 0.47026116]]], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([[[0.2886383,  0.27408165, 0.27809834]]], dtype=np.float32).reshape(1, 1, 3)
        self.class_name = LoadImagesAndLabels.class_name

        self.down_ratio = 4
        self.K = 100
        self.vis_thresh = 0.3
        self.show = True

    def pre_process(self, image):
        height, width = image.shape[0:2]
        inp_height, inp_width = LoadImagesAndLabels.default_resolution#获取分辨率
        torch.cuda.synchronize()
        s1 = time.time()
        inp_image = letterbox(image, height=inp_height)# 非形变图像pad

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        torch.cuda.synchronize()
        s2 = time.time()
        print("pre_process:".format(s2 -s1))
        meta = {'c': c, 's': s, 'out_height': inp_height // self.down_ratio, 'out_width': inp_width // self.down_ratio}
        return images, meta

    def predict(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            s1 = time.time()
            output = self.model(images)[-1]
            torch.cuda.synchronize()
            s2 = time.time()
            for k, v in output.items():
                print("output:", k, v.size())
            print("inference time:", s2 - s1)
            hm = output['hm'].sigmoid_()
            wh = output['wh']

            reg = output['reg'] if "reg" in output else None
            dets = ctdet_decode(hm, wh, reg=reg, K=self.K)
            torch.cuda.synchronize()
            return output, dets

    def post_process(self, dets, meta, scale=1):
        torch.cuda.synchronize()
        s1 = time.time()
        dets = dets.cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets, [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        torch.cuda.synchronize()
        s2 = time.time()
        print("post_process:", s2-s1)

        return dets[0]

    def work(self, image):
        img_h, img_w = image.shape[0], image.shape[1]
        torch.cuda.synchronize()
        s1 = time.time()
        detections = []
        images, meta = self.pre_process(image)
        output, dets = self.predict(images)
        hm = output['hm']
        dets = self.post_process(dets, meta)
        detections.append(dets)

        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

        final_result = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] >= self.vis_thresh:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    x1 = min(img_w, max(0, x1))
                    x2 = min(img_w, max(0, x2))
                    y1 = min(img_h, max(0, y1))
                    y2 = min(img_h, max(0, y2))
                    conf = bbox[4]
                    cls = self.class_name[j]
                    final_result.append((cls, conf, [x1, y1, x2, y2]))
        print("cost time: ", time.time() - s1)
        return final_result,hm

def demo(model_path,img_dir):
    output = "output"
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if LoadImagesAndLabels.num_classes <= 5:
        colors = [(255,155,50), (0,0, 255), (128,0,0), (255,0,255), (128,255,128), (255,0,0)]
    else:
        colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, LoadImagesAndLabels.num_classes + 1)][::-1]
    detector = CtdetDetector(model_path)

    img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
    print("image num:", len(img_list))
    for image_path in img_list:
        print("--------------------")
        img = cv2.imread(image_path)
        results,hm = detector.work(img)# 返回检测结果和置信度图
        print(results)
        class_num = {}
        for res in results:
            cls, conf, bbox = res[0], res[1], res[2]
            if cls in class_num:
                class_num[cls] += 1
            else:
                class_num[cls] = 1
            color = colors[LoadImagesAndLabels.class_name.index(cls)]
            # 绘制目标框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # 绘制标签&置信度
            txt = '{}:{:.1f}'.format(cls, conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.namedWindow("heatmap", 0)
        cv2.imshow("heatmap", np.hstack(hm[0].cpu().numpy()))
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == 27:
           break


if __name__ == '__main__':
    model_path = './model_save/model_hand_last.pth'# 模型路径
    img_dir = './example_hand/'# 测试集
    demo(model_path,img_dir)
