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
import json
import matplotlib.pyplot as plt
from data_iterator import LoadImagesAndLabels
from models.decode import ctdet_decode
from utils.model_utils import load_model
from utils.post_process import ctdet_post_process
from msra_resnet import get_pose_net as resnet
from xml_writer import PascalVocWriter

class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NpEncoder, self).default(obj)

# reference https://zhuanlan.zhihu.com/p/60707912
def draw_pr(coco_eval, label="192_288"):
    pr_array1 = coco_eval.eval["precision"][0, :, 0, 0, 2]
    score_array1 = coco_eval.eval['scores'][0, :, 0, 0, 2]

    x = np.arange(0.0, 1.01, 0.01)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True)

    plt.plot(x, pr_array1, "b-", label=label)
    for i in range(len(pr_array1)):
        print("Confidence: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(score_array1[i], pr_array1[i], x[i]))
    plt.legend(loc="lower left")
    plt.savefig("one_p_r.png")

def write_bbox_label(writer_x,img_shape,bbox,label):
	h,w = img_shape
	x1,y1,x2,y2 = bbox
	x1 = min(w, max(0, x1))
	x2 = min(w, max(0, x2))
	y1 = min(h, max(0, y1))
	y2 = min(h, max(0, y2))
	writer_x.addBndBox(int(x1), int(y1), int(x2), int(y2), label, 0)

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

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return keep
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
class CtdetDetector(object):
    def __init__(self,model_arch,model_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.num_classes = LoadImagesAndLabels.num_classes
        print('Creating model...')

        head_conv_ =64
        if "resnet_" in model_arch:
            num_layer = int(model_arch.split("_")[1])
            self.model = resnet(num_layers=num_layer, heads={'hm': self.num_classes, 'wh': 2, 'reg': 2}, head_conv=head_conv_, pretrained=False)  # res_18
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
        # print("pre_process:".format(s2 -s1))
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
            # for k, v in output.items():
            #     print("output:", k, v.size())
            # print("inference time:", s2 - s1)
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
        # print("post_process:", s2-s1)

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
        # print("cost time: ", time.time() - s1)
        return final_result,hm
def eval(model_arch,model_path,img_dir,gt_annot_path):
	output = "output"
	if os.path.exists(output):
		shutil.rmtree(output)
	os.mkdir(output)
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	if LoadImagesAndLabels.num_classes <= 5:
		colors = [(55,55,250), (255,155,50), (128,0,0), (255,0,255), (128,255,128), (255,0,0)]
	else:
		colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, LoadImagesAndLabels.num_classes + 1)][::-1]
	detector = CtdetDetector(model_arch,model_path)

	print('\n/****************** Eval ****************/\n')
	import tqdm
	import pycocotools.coco as coco
	from pycocotools.cocoeval import COCOeval

	print("gt path: {}".format(gt_annot_path))
	result_file = '../evaluation/instances_det.json'
	coco = coco.COCO(gt_annot_path)
	images = coco.getImgIds()
	num_samples = len(images)
	print('find {} samples in {}'.format(num_samples, gt_annot_path))
	#------------------------------------------------
	coco_res = []
	f = open("result.txt", "w")
	for index in tqdm.tqdm(range(num_samples)):
		img_id = images[index]
		file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
		image_path = os.path.join(img_dir, file_name)
		img = cv2.imread(image_path)
		results,hm = detector.work(img)# 返回检测结果和置信度图
		if len(results) == 0:
			f.write(os.path.basename(image_path) + "\n")

		class_num = {}
		for res in results:
			cls, conf, bbox = res[0], res[1], res[2]
			f.write(" ".join([file_name, cls, str(conf), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]) + "\n")
			coco_res.append({'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], 'category_id':
				LoadImagesAndLabels.class_name.index(cls), 'image_id': img_id, 'score': conf})
			if cls in class_num:
				class_num[cls] += 1
			else:
				class_num[cls] = 1
			color = colors[LoadImagesAndLabels.class_name.index(cls)]

			# 绘制标签&置信度

			label_ = '{}:{:.1f}'.format(cls, conf)
			plot_one_box(bbox, img, color=color, label=label_, line_thickness=2)

		cv2.imwrite(output + "/" + os.path.basename(image_path), img)
		cv2.namedWindow("heatmap", 0)
		cv2.imshow("heatmap", np.hstack(hm[0].cpu().numpy()))
		cv2.namedWindow("img", 0)
		cv2.imshow("img", img)
		key = cv2.waitKey(1)
	#-------------------------------------------------
	f.close()
	with open(result_file, 'w') as f_dump:
		json.dump(coco_res, f_dump, cls=NpEncoder)

	cocoDt = coco.loadRes(result_file)
	cocoEval = COCOeval(coco, cocoDt, 'bbox')
	# cocoEval.params.imgIds  = imgIds
	cocoEval.params.catIds = [1] # 1代表’Hand’类，你可以根据需要增减类别
	cocoEval.evaluate()
	print('\n/***************************/\n')
	cocoEval.accumulate()
	print('\n/***************************/\n')
	cocoEval.summarize()
	draw_pr(cocoEval)

def inference(model_arch,nms_flag,model_path,img_dir):
	print('\n/****************** Demo ****************/\n')
	flag_write_xml = False
	path_det_ = './det_xml/'
	if os.path.exists(path_det_):
		shutil.rmtree(path_det_)
	print('remove detect document ~')
	if not os.path.exists(path_det_):
		os.mkdir(path_det_)

	output = "output"
	if os.path.exists(output):
		shutil.rmtree(output)
	os.mkdir(output)
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	if LoadImagesAndLabels.num_classes <= 5:
		colors = [(55,55,250), (255,155,50), (128,0,0), (255,0,255), (128,255,128), (255,0,0)]
	else:
		colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, LoadImagesAndLabels.num_classes + 1)][::-1]
	detector = CtdetDetector(model_arch,model_path)
	for file_ in os.listdir(img_dir):
		if '.xml' in file_:
			continue
		print("--------------------")
		img = cv2.imread(img_dir + file_)
		if flag_write_xml:
			shutil.copyfile(img_dir + file_,path_det_+file_)
		if flag_write_xml:
			img_h, img_w = img.shape[0],img.shape[1]
			writer = PascalVocWriter("./",file_, (img_h, img_w, 3), localImgPath="./", usrname="RGB_HandPose_EVAL")
		results,hm = detector.work(img)# 返回检测结果和置信度图
		print('model_arch - {} : {}'.format(model_arch,results))
		class_num = {}
		nms_dets_ = []
		for res in results:
			cls, conf, bbox = res[0], res[1], res[2]
			if flag_write_xml:
				write_bbox_label(writer,(img_h,img_w),bbox,cls)
			if cls in class_num:
				class_num[cls] += 1
			else:
				class_num[cls] = 1
			color = colors[LoadImagesAndLabels.class_name.index(cls)]

			nms_dets_.append((bbox[0], bbox[1],bbox[2], bbox[3],conf))
			# 绘制标签&置信度
			if nms_flag == False:
				label_ = '{}:{:.1f}'.format(cls, conf)
				plot_one_box(bbox, img, color=color, label=label_, line_thickness=2)
		if flag_write_xml:
			writer.save(targetFile = path_det_+file_.replace('.jpg','.xml'))
		if nms_flag and len(nms_dets_)>0:
			#nms
			keep_ = py_cpu_nms(np.array(nms_dets_), thresh=0.8)
			print('keep_ : {}'.format(keep_))
			for i in range(len(nms_dets_)):
				if i in keep_:
					bbox_conf = nms_dets_[i]
					bbox_ = int(bbox_conf[0]),int(bbox_conf[1]),int(bbox_conf[2]),int(bbox_conf[3])
					label_ = 'nms_Hand:{:.2f}'.format(bbox_conf[4])
					plot_one_box(bbox_, img, color=(55,125,255), label=label_, line_thickness=2)


		cv2.namedWindow("heatmap", 0)
		cv2.imshow("heatmap", np.hstack(hm[0].cpu().numpy()))
		cv2.namedWindow("img", 0)
		cv2.imshow("img", img)
		key = cv2.waitKey(1)
		if key == 27:
			break

if __name__ == '__main__':
	model_arch = 'resnet_18'
	model_path = './model_save/model_hand_last_'+model_arch+'.pth'# 模型路径
	gt_annot_path = './hand_detect_gt.json'
	img_dir = '../done/'# 测试集

	nms_flag = True

	Eval = False

	if Eval:
		eval(model_arch,model_path,img_dir,gt_annot_path)
	else:
		inference(model_arch,nms_flag,model_path,img_dir)
