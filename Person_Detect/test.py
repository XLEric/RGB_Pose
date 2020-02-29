import argparse
import json
import time

from torch.utils.data import DataLoader

from utils.parse_config import parse_data_cfg
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import select_device

from yolov3 import Yolov3, Yolov3Tiny

def test(
        cfg,
        data_cfg,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.5,
        model=None
):
    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid']  # path to test images
    names = load_classes(data_cfg['names'])  # class names

    if model is None:
        device = select_device()
        num_classes = nc
        # Initialize model
        if "-tiny" in cfg:
            model = Yolov3Tiny(num_classes).to(device)
            # weights = 'weights-yolov3-tiny/best.pt'
            weights = "./yolov3-tiny_coco.pt"
        else:
            model = Yolov3(num_classes).to(device)
            # weights = 'weights-yolov3/best.pt'
            weights = "./finetune-weight/yolov3_coco.pt"

        # Load weights
        model.load_state_dict(torch.load(weights, map_location=device)['model'])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
    print("using device: {}".format(device))
    # Dataloader
    dataset = LoadImagesAndLabels(test_path, batch_size, img_size=img_size,augment = False)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        nt = len(targets)
        if nt == 0:  # if no targets continue
            continue
        imgs = imgs.to(device)
        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Build targets
        target_list = build_targets(model, targets)

        # Compute loss
        loss_i, _ = compute_loss(train_out, target_list)
        loss += loss_i.item()

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            correct, detected = [], []
            tcls = torch.Tensor()
            seen += 1

            if pred is None:
                if len(labels):
                    tcls = labels[:, 0].cpu()  # target classes
                    stats.append((correct, torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary

            if len(labels):
                # Extract target boxes as (x1, y1, x2, y2)
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
                tcls = labels[:, 0]  # target classes

                for *pbox, pconf, pcls_conf, pcls in pred:
                    if pcls not in tcls:
                        correct.append(0)
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)
            else:
                # If no labels add number of detections as incorrect
                correct.extend([0] * len(pred))

            # Append Statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls.cpu()))

    # Compute statistics
    stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
    nt = np.bincount(stats_np[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats_np):
        p, r, ap, f1, ap_class = ap_per_class(*stats_np)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

    # Print results per class
    if nc > 1 and len(stats_np):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    return mp, mr, map, mf1, loss
