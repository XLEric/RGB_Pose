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
### Train
1、Person Detect  
* [train datasets : coco2017/train2017 of person change to voc format(Baiduyun Password: )]()  
* [Pre-trained model (Baiduyun Password: t80y) ](https://pan.baidu.com/s/1QFAKGIv1zAgDLRyJej8SJA)  
* cd ./Person_Detect  
* Set the train parameters in " Person_Detect/cfg/voc_person.data "
* python train.py  

### Predict  

## Notice  
