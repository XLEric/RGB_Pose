#-*-coding:utf-8-*-
# date:2020-03-02
# Author: X.li
# function: train CenterNet only support resnet backbone

import os
import torch
import torch.utils.data
from opts import opts
from data_iterator import LoadImagesAndLabels
from utils.model_utils import load_model, save_model
from ctdet_trainer import CtdetTrainer
from msra_resnet import get_pose_net as resnet

def main(opt):
    path_save_model_ = './model_save/'
    if not os.path.exists(path_save_model_):
        os.mkdir(path_save_model_)

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    # opt = opts().update_dataset_info_and_set_heads(opt, LoadImagesAndLabels)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = torch.cuda.is_available()
    device_ = torch.device('cuda' if cuda else 'cpu')
    opt.device = device_
    chunk_sizes_ = [8]
    gpus_ = [0]
    # resnet_18 ,resnet_34 ,resnet_50,resnet_101,resnet_152
    model_arch = 'resnet_18'
    print('Creating model...')
    head_conv_ = 64

    num_layer = int(model_arch.split("_")[1])
    num_classes = 1
    heads_ = {'hm': num_classes, 'wh': 2, 'reg': 2}

    print('heads : {}'.format(heads_))
    model = resnet(num_layers=num_layer, heads=heads_,  head_conv=head_conv_, pretrained=True)  # res_18
    # print(model)


    batch_size_ = 16
    num_workers_ = 4
    learning_rate_ = 1.25e-4
    path_load_model_ = './model_save/model_hand_last_'+model_arch+'.pth'

    # path_load_model_ = ''
    lr_step_ = [190,220]

    optimizer = torch.optim.Adam(model.parameters(), learning_rate_)
    start_epoch = 0
    if os.path.exists(path_load_model_):
        model, optimizer, start_epoch = load_model(model, path_load_model_, optimizer, True, learning_rate_, lr_step_)

    trainer = CtdetTrainer(opt, model, optimizer)

    trainer.set_device(gpus_, chunk_sizes_, device_)

    print('load train_dataset')
    train_dataset = LoadImagesAndLabels(state = 'train',path_ = '../done/')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=num_workers_,
        pin_memory=False,
        drop_last=True
        )
    print('\n/***********************************/\n')
    print('Starting training...')
    print("using arch      : {}".format(model_arch))
    print('num_classes     : {}'.format(num_classes))
    print('batch_size      : {}'.format(batch_size_))
    print('num_workers     : {}'.format(num_workers_))
    print('learning_rate   : {}'.format(learning_rate_))
    print('lr_step         : {}'.format(lr_step_))
    print('path_load_model : {}'.format(path_load_model_))
    print('dataset len     : {}'.format(train_dataset.__len__()))

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, train_loader)

        save_model(path_save_model_ + 'model_hand_last_'+model_arch+'.pth', epoch, model, optimizer)
        if epoch%1==0:
            save_model(path_save_model_ + 'hand_'+model_arch+'_epoch_'+'{}.pth'.format(epoch), epoch, model, optimizer)

        if epoch in lr_step_:
            save_model(path_save_model_ + 'model_hand_{}.pth'.format(epoch), epoch, model, optimizer)
            lr = learning_rate_ * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
