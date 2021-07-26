# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torchvision
import datetime
import cv2
import os
import sys
import argparse
from JoTools.detectionResult import DeteObj, DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
import shutil
import numpy as np


def args_parse():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", "--img_dir", default=r"../img", help="")
    ap.add_argument("-ip", "--img_path", default=r"../img/test.jpg", help="")
    ap.add_argument("-am", "--assign_model", default=r"./diy_fas_2.pth", help="")
    ap.add_argument("-gpu", "--gpuID", type=str, default="2", help="")
    ap.add_argument("-conf", "--conf_th", type=float, default="0.35", help="")
    ap.add_argument("-sd", "--save_dir", type=str, default="./res", help="")
    assign_args = vars(ap.parse_args())  # vars 返回对象object的属性和属性值的字典对象
    return assign_args


def classify_one_img(assign_img_path, assign_save_folder, label_list):
    """检测一张图片"""
    # src_img = cv2.imread(assign_img_path)
    src_img = cv2.imdecode(np.fromfile(assign_img_path, dtype=np.uint8), 1)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = cv2.resize(src_img, (224,224))
    # cv2 读取后是 BGR 排列的，需要先转为 RBG
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
    img_tensor.to(device)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    out = model(img_tensor)
    pred = out.data.max(1, keepdim=True)[1]
    pre = pred.data.item()
    # fixme 获得置信度
    prob = torch.nn.functional.softmax(out.data, 1).max().item()

    save_label_folder = os.path.join(assign_save_folder, str(label_list[pre]))
    if not os.path.exists(save_label_folder): os.makedirs(save_label_folder)
    img_name = os.path.split(assign_img_path)[1]
    save_path = os.path.join(save_label_folder, img_name)
    shutil.copy(assign_img_path, save_path)


def save_test_log(train_log_folder):
    """记录训练命令"""
    if not os.path.exists(train_log_folder): os.makedirs(train_log_folder)
    train_log_path = os.path.join(train_log_folder, 'test_log.txt')
    time_str = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=13), "%Y-%m-%d-%H:%M:%S")
    with open(train_log_path, 'a') as txt_file:
        txt_file.write(time_str + " : ")
        txt_file.write("python3 ")
        txt_file.write(" ".join(sys.argv))
        txt_file.write("\n")



if __name__ == "__main__":

    # todo 分类模型直接将图片移动到指定的文件夹中，同一个文件夹中就是同一类
    # todo 使用 dataloader 一次性预测多个图片这样会快一些进行验证

    # python3 test.py -am ./model/test.pth -id ./imgs -gpu 2 -save ./res

    args = args_parse()
    save_test_log("./logs")
    # ----------------------------------------------------------------------------------------------------------------------
    model_path = args['assign_model']
    img_path = args['img_path']
    img_folder = args['img_dir'].rstrip('/')
    conf_th = float(args['conf_th'])
    save_folder = args['save_dir']
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpuID']
    # bg 是背景
    label_list = ["fzc_yt", "fzc_sm", "fzc_gt", "fzc_other", "zd_yt", 'zd_sm', "zd_gt", "zd_other", "qx_yt", "qx_sm", "qx_gt", "other"]
    # ----------------------------------------------------------------------------------------------------------------------
    img_folder = r"C:\Users\14271\Desktop\del\test\test"
    model_path = r"C:\Algo\Hand_CNN\FasterRcnn_pytorch\models\crop_18_1674.pth"
    save_folder = r"C:\Users\14271\Desktop\del\res"
    device = torch.device('cpu')
    # ----------------------------------------------------------------------------------------------------------------------

    model = torch.load(model_path)
    # model.cuda()
    model.to(device)
    model.eval()

    if os.path.isdir(img_folder):
        img_path_list = FileOperationUtil.re_all_file(img_folder, lambda x:str(x).endswith(('.jpg', '.JPG', '.png')))
        img_count = len(img_path_list)
        for img_index, each_img in enumerate(img_path_list):
            print_str = "{0}/{1} : {2}".format(img_index, img_count, each_img)
            print(print_str)
            classify_one_img(each_img, save_folder, label_list)
        # todo 计算每一个要素的准确率和召回率

    else:
        if os.path.isdir(img_path):
            print(img_path)
            classify_one_img(img_path, save_folder, label_list)
        else:
            print("* 未发现需要检测的图片")
            print("img folder : {0}".format(img_folder))
            print("img path : {0}".format(img_path))



























