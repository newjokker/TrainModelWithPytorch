# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import math
import sys
import time
import torch
from  . import utils
import torchvision.models.detection.mask_rcnn
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.operateDeteRes import OperateDeteRes
from JoTools.txkjRes.deteObj import DeteObj
from JoTools.utils.NumberUtil import NumberUtil
import prettytable

# 参考 :  https://github.com/pytorch/vision/tree/master/references/detection

"""
target 是一个 list 其中的每一个元素是字典，具体 type 如下:

{'boxes': tensor([[586.9481, 259.2025, 866.3922, 319.0185],
        [163.7322, 384.6138, 181.3783, 408.0201],
        [236.9199, 416.9780, 264.4015, 438.3615],
        [159.9716, 446.7415, 178.1962, 466.6802],
        [474.9970, 336.3565, 498.7179, 360.6296],
        [223.0345, 369.2986, 247.9126, 387.7924],
        [288.7010, 375.9449, 315.8932, 403.3966],
        [365.6493, 358.3179, 390.8166, 386.6366],
        [108.4798, 444.7188, 127.5723, 462.9236],
        [150.1361, 421.0236, 169.2285, 434.3160],
        [203.9421, 366.1200, 222.7452, 377.9676]], device='cuda:0'), 'labels': tensor([13, 17, 17, 17, 12, 16, 16, 16, 16, 16, 16], device='cuda:0'), 'image_id': tensor([1544], device='cuda:0'), 'area': tensor([199962.,   4941.,   7030.,   4347.,   6888.,   5504.,   8930.,   8526.,
          4158.,   3036.,   2665.], device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')}
"""

log_index = 0


def save_train_log(metric_logger, train_log_path):
    """保存训练日志"""
    global log_index
    log_index += 1
    lr = metric_logger.meters['lr']
    loss = metric_logger.meters['loss']
    loss_classifier = metric_logger.meters['loss_classifier']
    loss_box_reg = metric_logger.meters['loss_box_reg']
    loss_objectness = metric_logger.meters['loss_objectness']
    loss_rpn_box_reg = metric_logger.meters['loss_rpn_box_reg']

    with open(train_log_path, 'a') as train_log_file:
        train_log_file.write("index :               {0},\n".format(log_index))
        train_log_file.write("lr :                  {0},\n".format(lr))
        train_log_file.write("loss :                {0},\n".format(loss))
        train_log_file.write("loss_classifier :     {0},\n".format(loss_classifier))
        train_log_file.write("loss_box_reg :        {0},\n".format(loss_box_reg))
        train_log_file.write("loss_objectness :     {0},\n".format(loss_objectness))
        train_log_file.write("loss_rpn_box_reg :    {0},\n".format(loss_rpn_box_reg))
        train_log_file.write("-" * 50 + '\n')

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, train_log_path='./train.log'):
    """训练一个 epoch，返回训练信息"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    index = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # fixme 打印 loss
        index += 1
        # print("{0} loss : ".format(index), losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            # fixme 这边是不是需要退出，不能 continue 吗
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # save log
        if index % print_freq == 0:
            save_train_log(metric_logger, train_log_path)

    return metric_logger

def train_one_epoch_classify(model, optimizer, data_loader, epoch, device, print_loss=50):
    """训练一个 epoch，返回训练信息"""
    model.train()
    loss_function=nn.CrossEntropyLoss()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    index = 0
    sum_loss = 0
    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        #
        predict = model(images)
        losses = loss_function(predict, targets)
        sum_loss += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        index += 1
        if math.fmod(index, print_loss) == 0:
            print(index, sum_loss)
            sum_loss = 0

def train_one_epoch_segment(model, optimizer, data_loader, device, epoch, print_freq):
    """分割训练一个epoch"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

# ----------------------------------------------------------------------------------------------------------------------

def _target_to_deteres(target, is_tensor=True, label_dict=None):
    """将 target 转为 deteres"""

    if is_tensor:
        boxes = target['boxes'].cpu().data.numpy().tolist()
        labels = target['labels'].cpu().data.numpy().tolist()
        # 处理没有 score 的情况
        if 'scores' in target:
            scores = target['scores'].cpu().data.numpy().tolist()
        else:
            scores = [-1] * len(boxes)
    else:
        boxes = target['boxes']
        labels = target['labels']
        if 'scores' in target:
            scores = target['scores']
        else:
            scores = [-1] * len(boxes)
    # 得到的结果
    dete_res = DeteRes()
    for i in range(len(boxes)):
        each_box = boxes[i]
        # lable 数字转为 tag
        if label_dict:
            each_label = label_dict[int(labels[i])]
        else:
            each_label = labels[i]
        each_score = scores[i]
        x1, y1, x2, y2 = each_box
        dete_res.add_obj(x1, y1, x2, y2, tag=each_label, conf=each_score)
    return dete_res

def _dict_add(dict_1, dict_2):
    """字典之间的相加"""
    res = dict_1.copy()
    # 整合
    for each_key in dict_2:
        if each_key in res:
            res[each_key] += dict_2[each_key]
        else:
            res[each_key] = dict_2[each_key]
    return res

def print_evaluate_res(res_conf_dict=None, acc_conf_rec=None, label_dict=None):
    """将结果进行打印"""
    stand_label_list = list(label_dict.values())
    # ------------------------------------------------------------------------------------
    for conf in acc_conf_rec:
        tb_1 = prettytable.PrettyTable()
        tb_1.field_names = ["class", "conf", "acc", "rec"]
        acc_rec = acc_conf_rec[conf]
        for each_tag in stand_label_list:
            each_acc = NumberUtil.format_float(acc_rec[each_tag]['acc'], 3)
            each_rec = NumberUtil.format_float(acc_rec[each_tag]['rec'], 3)
            tb_1.add_row([each_tag, conf, each_acc, each_rec])
        print(tb_1)

    # ------------------------------------------------------------------------------------

    for conf in acc_conf_rec:
        res_dict = res_conf_dict[conf]
        tb_2 = prettytable.PrettyTable()
        tb_2.field_names = [" ", "conf", "num"]
        #
        for assign_mode in ['correct', 'miss', 'extra']:
            for each_tag in stand_label_list:
                assign_tag = "{0}_{1}".format(assign_mode, each_tag)
                if assign_tag in res_dict:
                    tb_2.add_row([assign_tag, conf, res_dict[assign_tag]])

        # 添加 mistake
        for i in stand_label_list:
            for j in stand_label_list:
                mistake_str = "mistake_{0}-{1}".format(i, j)
                if mistake_str in res_dict:
                    tb_2.add_row([mistake_str, conf, res_dict[mistake_str]])

        print(tb_2)

def model_performance_index(acc_conf_rec, assign_label_list):
    """计算模型性能指数"""
    # 各个 conf 所有 tag 的平均 res 和 acc
    res_acc_list = []
    for each_conf in acc_conf_rec:
        each_acc_rec = acc_conf_rec[each_conf]
        for each_tag in assign_label_list:
            if each_tag in each_acc_rec:
                each_acc = each_acc_rec[each_tag]['acc']
                each_rec = each_acc_rec[each_tag]['rec']
                res_acc_list.append(max(each_acc, 0))
                res_acc_list.append(max(each_rec, 0))
    return np.mean(res_acc_list)

@torch.no_grad()
def evaluate(model, data_loader, device, label_dict=None, conf_list=None, do_print=True):
    """验证"""

    # 对每个精度下的结果进行计算
    if conf_list is None:
        conf_list = [0.4,0.5,0.6,0.7,0.8,0.9]

    cpu_device = torch.device("cuda")
    model.eval()
    #
    a = OperateDeteRes()
    a.iou_thershold = 0.4       # 重合度阈值
    res_conf_dict = {}
    acc_conf_rec = {}
    for conf in conf_list:
        print("cal conf : {0}".format(conf))
        res_dict = {}
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            # 参考 ：https://blog.csdn.net/u013548568/article/details/81368019
            torch.cuda.synchronize(device)
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # 每一张图片进行对比
            for i in range(len(outputs)):
                res = _target_to_deteres(outputs[i], label_dict=label_dict)
                real = _target_to_deteres(targets[i], label_dict=label_dict)
                res.filter_by_conf(conf)
                rere = a.compare_customer_and_standard(real, res)
                res_dict = _dict_add(res_dict, rere)
        acc_rec = a.cal_acc_rec(res_dict, tag_list=list(label_dict.values()))
        res_conf_dict[conf] = res_dict
        acc_conf_rec[conf] = acc_rec
    # 打印模型性能具体参数
    if do_print:
        print_evaluate_res(res_conf_dict, acc_conf_rec, label_dict)
    # 计算模型性能指数
    model_pd = model_performance_index(acc_conf_rec, list(label_dict.values()))
    print("model_pd : {0}".format(model_pd))
    return model_pd

# ----------------------------------------------------------------------------------------------------------------------

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate_classify(model, data_loader, device):

    # todo 参考下面的进行编写

    """
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    """


    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # cpu_device = torch.device("cpu")
    cpu_device = torch.device("cuda")
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


