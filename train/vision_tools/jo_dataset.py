# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import torch
import numpy as np
from PIL import Image
from JoTools.txkj.parseXml import parse_xml
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.segmentJson import SegmentJson
from JoTools.txkjRes.deteRes import DeteRes


class DeteDataset(torch.utils.data.Dataset):
    """解析数据，得到符合规范的 dataset"""

    def __init__(self, root, label_dict, assign_transforms=None):
        self.root_dir = root
        self.label_dict = label_dict
        self.transforms = assign_transforms
        self.img_max_szie = 1200
        #
        self.imgs, self.xmls = [], []
        xml_dir = os.path.join(root, "Annotations")
        img_dir = os.path.join(root, "JPEGImages")
        #
        print("* check data ")
        obj_num_dict = {}
        for each_xml_path in FileOperationUtil.re_all_file(xml_dir, endswitch=['.xml']):
            each_img_path = os.path.join(img_dir, FileOperationUtil.bang_path(each_xml_path)[1] + '.jpg')
            # filter img
            if os.path.exists(each_img_path):
                dete_xml = DeteRes(each_xml_path)
                obj_num_dict = self._dict_add(dete_xml.count_tags(), obj_num_dict)
                # filter xml
                if len(dete_xml) > 0:
                    self.imgs.append(each_img_path)
                    self.xmls.append(each_xml_path)
        # print
        print('-' * 50)
        for each_item in obj_num_dict.items():
            print(each_item)
        print('-' * 50)

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root_dir, "JPEGImages", self.imgs[idx])
        xml_path = os.path.join(self.root_dir, "Annotations", self.xmls[idx])
        #
        img = Image.open(img_path).convert("RGB")
        # get resize ratio
        resize_ratio = None
        # height, width = img.height, img.width
        # if max(height, width) < self.img_max_szie:
        #     # fixme do not resize ?
        #    resize_ratio = None
        # else:
        #     resize_ratio = self.img_max_szie / max(height, width)
        # 读取 xml 信息
        xml_info = parse_xml(xml_path)
        # --------------------------------------------------------------------------------------------------------------
        # target = xml_info_to_target(xml_info, self.label_dict, idx)
        boxes, labels = [], []
        for each_object in xml_info['object']:
            name = each_object['name']
            labels.append(np.int(self.label_dict[name]))
            bndbox = each_object['bndbox']
            xmin, ymin, xmax, ymax = float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])
            # resize target
            if resize_ratio is not None:
                # todo 实验一下这边是不是需要转为整数
                xmin, ymin, xmax, ymax = xmin*resize_ratio, ymin*resize_ratio, xmax*resize_ratio, ymax*resize_ratio
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(xml_info['object']),), dtype=torch.int64)
        #
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        # --------------------------------------------------------------------------------------------------------------

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
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


class ClassifyDataset(torch.utils.data.Dataset):
    """获取分类模型的 dataset"""

    def __init__(self, root, label_list, assign_transforms=None):
        self.root_dir = root
        self.label_list = label_list
        self.transforms = assign_transforms
        self.imgs = []
        self.labels = []
        self.label_count_dict = {}

        # fixme 看看图片的 label_index 是不是需要从 1 开始，0 留给背景
        # 遍历所有的图片和其对应的 label，图片的级别不需要固定
        for label_index, each_label in enumerate(label_list):
            img_dir = os.path.join(self.root_dir, each_label)
            self.label_count_dict[each_label] = 0
            if not os.path.exists(img_dir):
                print("img_dir not exists : {0}".format(img_dir))
                continue
            for each_img_path in FileOperationUtil.re_all_file(img_dir, lambda x:str(x).endswith(('.jpg', '.JPG', '.png', '.PNG'))):
                self.imgs.append(each_img_path)
                self.labels.append(label_index)
                self.label_count_dict[each_label] += 1

        # 展示训练图片每个类型有多少张
        print('-'*50)
        for each in self.label_count_dict.items():
            print(each)
        print('-'*50)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224,224))

        target = torch.as_tensor(self.labels[item], dtype=torch.int64)
        #
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class SegmentDataset(torch.utils.data.Dataset):
    """获取分割模型的 dataset"""

    def __init__(self, root, label_dict, assign_transforms=None):
        """init"""
        self.root = root
        self.json_path_list = []
        self.transforms = assign_transforms
        #
        for each_json_path in FileOperationUtil.re_all_file(root, endswitch=['.json']):
            self.json_path_list.append(each_json_path)

    def __getitem__(self, idx):
        """解析 json 数据，获取其中的信息"""
        # 原数据是 json
        json_path = self.json_path_list[idx]
        segment_json = SegmentJson(json_path)
        segment_json.parse_json_info(parse_mask=True, parse_img=True)
        #
        img = segment_json.image_data
        mask = segment_json.mask
        #
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []

        # 获取 box
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.json_path_list)


