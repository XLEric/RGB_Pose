# RGB_HandPose
RGB_HandPose
## Introduction
The project consists of three parts.  
1、Person Detect  
example(Person Detect):  
![person](https://github.com/XiangLiK/RGB_HandPose/raw/master/samples/person.png)  
2、Hand Detect  
example(Hand Detect):  
![hand](https://github.com/XiangLiK/RGB_HandPose/raw/master/samples/hand.png)   
3、HandPose Estimation  
example(HandPose Estimation):  
![hand](https://github.com/XiangLiK/RGB_HandPose/raw/master/samples/handpose.png)  

## Requirements  

* Python 3.6  
* PyTorch 1.1
* torchvision 0.3.0
* OpenCV 3.4.0

## Usage  
### Person Detect  
* Person Detect - YoloV3  
#### Train  
* [detect_person_datasets : coco2017/train2017 of person change to voc format(Baiduyun Password: x9u4 )](https://pan.baidu.com/s/1Z7RBbrmR9iRK61-RTy5E6A)  
* [Pre-trained model (Baiduyun Password: t80y) ](https://pan.baidu.com/s/1QFAKGIv1zAgDLRyJej8SJA)  
* cd ./Person_Detect  
* release detect_person_datasets
* change make_train_person_datasets.py 'path_data' for the path of detect_person_datasets
* make train datasets: python make_train_person_datasets.py
* Set the train parameters in " Person_Detect/cfg/voc_person.data "
* python train.py  

#### Predict  
* set predict.py params(  
  model_path : 检测模型路径
  images_path : 测试文件夹  
  model_cfg : 模型类型  
  voc_config : 模型相关配置文件  
  img_size : 图像尺寸    
  conf_thres : 物体检测置信度  
  nms_thres ：nms阈值)  
* run: python predict.py

### Hand Detect  
* Hand Detect - CenterNet , But only support resnet backbone  
#### Train  
* [detect_hand_datasets : it use person's image of crop from coco2017/train2017.It label for ourself, change to voc format(Baiduyun Password:)]()  
* release detect_hand_datasets  
* train.py ,change 'LoadImagesAndLabels' 'path_' for your path of the datasets  
* run:python train.py  
#### Predict
* predict.py ,change 'model_path' for your path of model ,change 'img_dir' for your path of test datasets
* run: python predict.py  

## Third-Party Resources  
* https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/blob/master/yolo/vedanet/network/backbone/brick/darknet53.py  
* https://github.com/ultralytics/yolov3/blob/master/models.py  
* https://github.com/xingyizhou/CenterNet  

## Notice  
* I hope that'll be helpful for you.

## Contact  
* E-mails: 305141918@qq.com
