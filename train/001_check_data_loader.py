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



root_dir = r"/home/suanfa-4/ldq/001_train_data/fzc_step_1_resize"
label_dict = {"fzc":1, "other":2}


train_dataset = GetDataset(root_dir, label_dict, None)

print(len(train_dataset))

for each in train_dataset:
    print(each)
    break





