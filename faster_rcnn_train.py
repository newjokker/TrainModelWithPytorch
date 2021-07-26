# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import os
import cv2
import datetime
import sys
import argparse
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from vision_tools import transforms as T
from vision_tools.engine import train_one_epoch, evaluate
from vision_tools import utils
from vision_tools.jo_dataset import GetDataset
from JoTools.txkj.parseXml import parse_xml


"""
# 运行的 torch 环境
    * python 3.6.10 
    * torch-1.5.0+cu101-cp36-cp36m-linux_x86_64.whl
    * torchvision-0.6.0+cu101-cp36-cp36m-linux_x86_64.whl
    * python 3.5.6
    * torch-1.5.0+cu92-cp35-cp35m-linux_x86_64
    * torchvision-0.6.0+cu92-cp35-cp35m-linux_x86_64
    
# 当前代码验证部分是 cpu 跑的，所以会特别慢
# 参考：
"""

# todo 实现裁剪 transform
# todo transform 中每次训练将标签位置随机位移一些像素，一定的比例

"""
* python3 train.py -rd /home/ldq/000_train_data/wtx_fas_train_data --gpuID 2 -sd ./model -ep 300 -bs 5 -se 5 -mv 4 
"""
# fixme 为什么验证集的打破的数据每次都不一样
# todo 查看一下是不是因为 transform 才导致训练那么慢的


# todo 训练的时候需要对放进去的图片进行缩放，我之前训练慢就是因为没有缩放直接放进去了


def args_parse():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-rd", "--root_dir", type=str)
    ap.add_argument("-td", "--test_dir", type=str, default="")
    ap.add_argument("-gpu", "--gpuID", type=str, default="2", help="")
    ap.add_argument("-sd", "--save_dir", type=str, default="./models", help="")
    ap.add_argument("-sn", "--save_name", type=str, default=None, help="")
    ap.add_argument("-ep", "--epoch_num", type=int, default=300, help="")
    ap.add_argument("-bs", "--batch_size", type=int, default=5, help="")
    ap.add_argument("-am", "--assign_model", type=str, default=None, help="")
    ap.add_argument("-nw", "--num_workers", type=int, default=12, help="")
    ap.add_argument("-se", "--save_epoch", type=int, default=5, help="多少个 epoch 保存一次")
    ap.add_argument("-ae", "--add_epoch", type=int, default=0, help="增加的 epoch")
    ap.add_argument("-cl", "--class_list", type=str, default=None, help="分类类别")
    ap.add_argument("-log", "--train_log_path", type=str, default='./logs/train.log', help="训练日志地址")
    assign_args = vars(ap.parse_args())  # vars 返回对象object的属性和属性值的字典对象
    return assign_args

def get_transform(train):
    # converts the image, a PIL image, into a PyTorch Tensor
    # fixme 这边先将图像结构转为 numpy 处理后再转为 tensor
    # todo target 要做同样的处理，先处理为 tensor 难以操作
    assign_transforms = [T.ImageToNumpy()]

    if train:
        # 水平旋转
        assign_transforms.append(T.RandomHorizontalFlip(0.5))
        # 改变图像亮度
        assign_transforms.append(T.RandomChangeImgLight(0.5))
        # 改变通道顺序
        assign_transforms.append(T.RandomChangechannelOrder(0.5))
        # 增加噪声
        assign_transforms.append(T.AddGasussNoise(0.5))
        # 增加改变图像大小
        assign_transforms.append(T.RandomResize(0.5))

    # 转变为 tensor
    assign_transforms.append(T.ToTensor())

    return T.Compose(assign_transforms)

def print_log(metric_logger):
    """打印训练信息"""
    lr = metric_logger.meters['lr']
    loss = metric_logger.meters['loss']
    loss_classifier = metric_logger.meters['loss_classifier']
    loss_box_reg = metric_logger.meters['loss_box_reg']
    loss_objectness = metric_logger.meters['loss_objectness']
    loss_rpn_box_reg = metric_logger.meters['loss_rpn_box_reg']
    #
    print("lr : {0}".format(lr))
    print("loss : {0}".format(loss))
    print("loss_classifier : {0}".format(loss_classifier))
    print("loss_box_reg : {0}".format(loss_box_reg))
    print("loss_objectness : {0}".format(loss_objectness))
    print("loss_rpn_box_reg : {0}".format(loss_rpn_box_reg))
    print('-' * 50)

def save_train_log(train_log_folder):
    """记录训练命令"""
    if not os.path.exists(train_log_folder): os.makedirs(train_log_folder)
    train_log_path = os.path.join(train_log_folder, 'train_log.txt')
    time_str = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=13), "%Y-%m-%d-%H:%M:%S")
    with open(train_log_path, 'a') as txt_file:
        txt_file.write(time_str + " : ")
        txt_file.write("python3 ")
        txt_file.write(" ".join(sys.argv))
        txt_file.write("\n")


if __name__ == "__main__":

    args = args_parse()
    train_log_dir = "./logs"
    save_train_log(train_log_dir)
    # ----------------------------------------------------------------------------------------------------------------------
    root_dir = args["root_dir"].rstrip('/')
    test_dir = args["test_dir"].rstrip('/')
    device = torch.device('cuda')
    batch_size = args["batch_size"]
    num_epochs = args["epoch_num"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpuID"]
    # fixme num_works 多了之后就会报错，显示为内存不足，看看原因
    num_workers = args["num_workers"]
    save_dir = args["save_dir"]
    save_name = args["save_name"]
    save_epoch = args["save_epoch"]
    train_log_path = args["train_log_path"]

    if save_name is None:
        save_name = os.path.split(root_dir)[1]
    # ----------------------------------------------------------------------------------------------------------------------
    # label_list = ["fzc_yt", "fzc_sm", "fzc_gt", "fzc_other", "zd_yt", 'zd_sm', "zd_gt", "zd_other", "qx_yt", "qx_sm", "qx_gt", "other"]
    label_list = ["fzc", "other"]
    # label_list = list(map(lambda x: x.strip(), args["class_list"].split(',')))
    # ----------------------------------------------------------------------------------------------------------------------
    label_dict = {label_list[i]: i + 1 for i in range(len(label_list))}
    num_classes = len(label_list) + 1
    # ----------------------------------------------------------------------------------------------------------------------

    # get train_dataset, test_dataset
    if test_dir:
        # get dataset
        train_dataset = GetDataset(root_dir, label_dict, get_transform(train=True))
        dataset_test = GetDataset(test_dir, label_dict, get_transform(train=False))
    else:
        # get dataset
        train_dataset = GetDataset(root_dir, label_dict, get_transform(train=True))
        dataset_test = GetDataset(root_dir, label_dict, get_transform(train=False))
        # do test for 200 img
        indices = torch.randperm(len(train_dataset)).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:-200])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-200:])

    # get data_loader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  num_workers=num_workers, collate_fn=utils.collate_fn)

    # get model
    add_epoch = 0
    if args["assign_model"] is None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
        # torchvision.models.detection.
    else:
        model = torch.load(args["assign_model"])
        add_epoch = args["add_epoch"]

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

    # learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # training
    max_model_pd = 0
    for epoch in range(num_epochs):
        # update epoch
        epoch += add_epoch + 1
        # train for one epoch
        # print_freq = 50, 每 50 次进行一次打印
        each_metric_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, train_log_path=train_log_path)
        # save_log(each_metric_logger)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch % save_epoch == 0:
            # fixme 记录详细的验证日志
            model_pd = evaluate(model, data_loader_test, device=device, label_dict={i+1:label_list[i] for i in range(len(label_list))})
            if model_pd > max_model_pd:
                model_path = os.path.join(save_dir, "{0}_best.pth".format(save_name))
                torch.save(model, model_path)
        # save model
        if epoch % save_epoch == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, "{0}_{1}_{2}.pth".format(save_name, epoch, epoch*len(data_loader_train)))
            torch.save(model, model_path)


































