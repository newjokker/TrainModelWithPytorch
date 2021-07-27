# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import json
import configparser
import numpy as np
import os
import torch
import random
from Crypto.Cipher import AES
import struct

from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except, timeStamp
from ..detect_libs.abstractBase import detection
from torch.backends import cudnn



class FasterDetectionPytorch(detection):

    def __init__(self, args, objName=None, scriptName=None):
        super(FasterDetectionPytorch, self).__init__(objName, scriptName)
        self.encryption = False
        self.readArgs(args)
        self.readCfg()
        self.args = args
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        # 指定 GPU 的使用
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.debug = self.cf.getboolean("common", 'debug')
        # fixme 这边要和之前的规范进行统一
        self.tfmodelName = self.cf.get(self.objName, 'modelname')
        self.dataset = self.cf.get(self.objName, 'dataset')
        # self.anchorScales = eval(self.cf.get(self.objName, "anchorscales"))
        # self.anchorRatios = eval(self.cf.get(self.objName, "anchorsatios"))
        self.CLASSES = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName, 'visible_classes').strip(',').split(','))
        self.confThresh = self.cf.getfloat(self.objName, 'conf_threshold')
        self.iouThresh = self.cf.getfloat(self.objName, 'iou_threshold')
        # self.compoundCoef = int(self.cf.getfloat(self.objName, 'compound_coef'))
        # self.inputSizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536][self.compoundCoef]
        self.encryption = self.cf.getboolean("common", 'encryption')
        print(self.cf.getboolean("common", 'encryption'))

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型
        model_path = os.path.join(self.modelPath, self.tfmodelName)
        print("self.encryption:", self.encryption)
        print(model_path)

        # fixme 下面两个不注释的话，会占用大量的 gpu
        # cudnn.fastest = True
        # cudnn.benchmark = True
        self.net = torch.load(model_path)
        self.net.eval()
        self.net.cuda()
        self.warmUp()
        self.log.info('model restore end')

        # 删除解密的模型
        if self.encryption:
            self.log.info(model_path)
            self.delDncryptionModel(model_path)
        return

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((1000, 1000, 3), dtype=np.uint8)
        self.detect(im, 'warmup.jpg')

    def display(self, preds, imgs):
        """数据展示"""
        res = []
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = self.CLASSES[preds[i]['class_ids'][j]]
                score = preds[i]['scores'][j]
                res.append([obj, j, int(x1), int(y1), int(x2), int(y2), str(score)])
        return res

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg"):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        img_tensor = torch.from_numpy(im / 255.).permute(2, 0, 1).float().cuda()
        out = self.net([img_tensor])
        res = []
        # 结果处理并输出
        boxes, labels, scores = out[0]['boxes'], out[0]['labels'], out[0]['scores']

        index = 0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i].item()
            if score > self.confThresh:
                index += 1
                obj = self.CLASSES[labels[i].item()-1]
                res.append([obj, index, int(x1), int(y1), int(x2), int(y2), str(score)])

        # todo 这边加上条件过滤，nms


        return res

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        salt = 'txkj2019'
        bkey32 = "{: <32}".format(salt).encode("utf-8")
        suffixs = [".pth"]
        random_id = str(random.randint(1, 100))
        modelBasePath = self.modelPath
        for sf in suffixs:
            model_locked_name = os.path.splitext(self.tfmodelName)[0] + "_locked" + sf
            model_src_name = model_locked_name.replace("_locked", random_id)
            src_Fmodel = os.path.join(self.cachePath, model_src_name)
            dst_Fmodel = os.path.join(modelBasePath, model_locked_name)
            decrypt_file(bkey32, dst_Fmodel, src_Fmodel)

        ### 解密后的模型 ####
        tfmodel_SrcName = os.path.splitext(self.tfmodelName)[0] + random_id + '.pth'
        tfmodel = os.path.join(self.cachePath, tfmodel_SrcName)
        return tfmodel

    @try_except()
    def delDncryptionModel(self, tfmodel):
        suffixs = [".pth"]
        for sf in suffixs:
            model_src_name = tfmodel.replace(".pth", sf)
            # src_Fmodel = os.path.join(self.cachePath, model_src_name)
            delete_scrfile(model_src_name)
        self.log.info('delete ljc dncryption model successfully! ')
        return


def decrypt_file(key, in_filename, out_filename=None, chunksize=24 * 1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)


def delete_scrfile(srcfile_path):
    os.remove(srcfile_path)
