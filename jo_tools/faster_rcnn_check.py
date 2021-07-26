# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from JoTools.detectionResult import DeteObj, OperateDeteRes, DeteRes

# todo 计算两个 target 之间的差异


"""
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
        'labels': tensor([13, 17, 17, 17, 12, 16, 16, 16, 16, 16, 16], device='cuda:0'), 'image_id': tensor([1544], device='cuda:0'),
        'area': tensor([199962.,   4941.,   7030.,   4347.,   6888.,   5504.,   8930.,   8526., 4158.,   3036.,   2665.], device='cuda:0'),
        'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')},
        'scores':tensor[1,1,1,0.05]
"""

# todo 计算两个范围的 iou  (x1,y1,x2,y2)
# todo 最后的出来的是多检多少，漏检多少，错检多少，
# todo 可以指定到每个 label 的效果，这样说不定可以指定每个 label 最合适的 score
# todo 计算 每个等级的 conf 下的多检，错检，漏检数目
# todo 统计错检，漏检的基本信息，各个 label 的数目，面积占比等


def target_to_deteres(target, is_tensor=True):
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
        each_label = labels[i]
        each_score = scores[i]
        x1, y1, x2, y2 = each_box
        dete_res.add_obj(x1, y1, x2, y2, tag=each_label, conf=each_score)
    return dete_res

def dict_add(dict_1, dict_2):
    """字典之间的相加"""
    res = dict_1.copy()
    # 整合
    for each_key in dict_2:
        if each_key in res:
            res[each_key] += dict_2[each_key]
        else:
            res[each_key] = dict_2[each_key]
    return res

def format_dict(res_dict):
    """规范检测结果字典"""
    res = {'correct':0, 'extra':0, 'miss':0}
    for each in res_dict:
        if each.startswith('correct'):
            res['correct'] += res_dict[each]
        elif each.startswith('miss'):
            res['miss'] += res_dict[each]
        if each.startswith('extra'):
            res['extra'] += res_dict[each]
    return res







if __name__ == "__main__":

    # fixme 现在的问题是，test 完了之后不能直接显示正确率，各个 conf 下的正确率


    model_path = r""


    pass











































