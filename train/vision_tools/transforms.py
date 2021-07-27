import random
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import torch.nn.functional as nnf
from torchvision.transforms import transforms


# fixme 能将图片和 box 一起传入进行处理 albumentations

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
        [203.9421, 366.1200, 222.7452, 377.9676]], device='cuda:0'), 
        'labels': tensor([13, 17, 17, 17, 12, 16, 16, 16, 16, 16, 16], device='cuda:0'), 
        'image_id': tensor([1544], device='cuda:0'), 
        'area': tensor([199962.,   4941.,   7030.,   4347.,   6888.,   5504.,   8930.,   8526.,
          4158.,   3036.,   2665.], device='cuda:0'), 
          'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')}
"""


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = Image.fromarray(image)
        image = F.to_tensor(image)
        return image, target


class ImageToNumpy(object):
    def __call__(self, image, target):
        image = np.array(image)
        return image, target

# ----------------------------------------------------------------------------------------------------------------------

# fixme 输入的是 Image 结构，输出的 是 Image 结构,中间使用 numpy


class RandomHorizontalFlip(object):
    """水平方向旋转扩增"""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            image = np.flip(image, -1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomChangeImgLight(object):
    """改变图像明暗"""

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 设置增强系数为 0.5 - 1.5
            image = image * (random.randrange(50, 150) * 0.01)
            image = image.astype(np.uint8)
            image[image>255] = 255
        return image, target


class RandomChangechannelOrder(object):
    """改变图像通道的顺序"""

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # image = np.array(image)
            a, b = random.sample([0,1,2], 2)
            image[a,:,:], image[b,:,:] = image[b,:,:], image[a,:,:]
        return image, target


class RandomResize(object):
    """对图片随机进行下采样，1-3 倍"""

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # random resize ratio
            # image = np.array(image)
            img_ratio = random.randint(30, 100)*0.01
            height, width = image.shape[:2]
            # 将 tensor resize
            # print(image.shape)
            image = np.resize(image, (int(height*img_ratio), int(width*img_ratio), 3))
            # print(image.shape)
            # print('-'*50)
            # image = image.unsqueeze(0)
            # image = nnf.interpolate(image, size=(height, width))
            # image = image.squeeze(0)
            #
            for each_box in target["boxes"]:
                # print(each_box)
                # print(each_box.dtype)
                # fixme 这边要转为 整数，传过来的 box 位置是 负值
                each_box *= img_ratio
                each_box = each_box.type(torch.int64)
                each_box = each_box.type(torch.float)
                # print(img_ratio)
                # print(each_box)
                # print(each_box.dtype)
                # print('-'*20)
        return image, target


class AddGasussNoise(object):
    """增加高斯噪声"""

    def __init__(self, prob, mean=0, var=0.04):
        self.prob = prob
        self.mean = mean
        self.var = var

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 设置增强系数为 0.5 - 1.5
            image = np.array(image)
            image = image/255.0
            noise = np.random.normal(self.mean, self.var ** 0.5, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1.0)
            image = np.uint8(image * 255)
        return image, target


# ----------------------------------------------------------------------------------------------------------------------


class RandomVerticalFlip(object):
    """竖直方向旋转扩增"""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomClip(object):
    """裁剪图像"""

    """
    * 要是图像中的对象大于 两个
    * 从图像中随机选择两个对象，包含这两个对象的最小范围即为新的图像范围
    * 计算其他标注是否在这个范围内，在范围内，就增加这个要素的 target ，否者去除
    """


    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 设置增强系数为 0.5 - 1.5
            image = image/255.0
            noise = np.random.normal(mean, var ** 0.5, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1.0)
            image = np.uint8(image * 255)
        return image, target


# todo 增加一个 test_img 和 test_xml 进行测试

