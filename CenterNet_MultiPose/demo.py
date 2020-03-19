from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    idx_ = 0
    while True:
        _, img = cam.read()
        idx_ += 1
        if idx_%2==0:
            continue
        # cv2.namedWindow('input',0)
        # cv2.imshow('input', img)
        img_,ret = detector.run(img)

        cv2.putText(img,'MultiPose fps:{:.2f}'.format(1./ret['tot']),(10,img.shape[0]-10),font,2.0,(185,55,255),12)
        cv2.putText(img,'MultiPose fps:{:.2f}'.format(1./ret['tot']),(10,img.shape[0]-10),font,2.0,(185,255,55),3)

        img_s = np.hstack((img,img_))
        cv2.line(img_s,(int(img_s.shape[1]/2),0),(int(img_s.shape[1]/2),int(img_s.shape[0])),(25,90,255),20)
        cv2.line(img_s,(int(img_s.shape[1]/2),0),(int(img_s.shape[1]/2),int(img_s.shape[0])),(120,120,120),8)

        cv2.namedWindow('img_s',0)
        cv2.imshow('img_s', img_s)

        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        # print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    print('------------->>.')
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    for (image_name) in image_names:

      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      # print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  try:
      demo(opt)
  except:
      pass
