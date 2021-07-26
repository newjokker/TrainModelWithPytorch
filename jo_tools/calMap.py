# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import  torch


# todo 计算 Map 值


# class CalMap():
#
#     def __init__(self):
#         self.model = None
#         self.val_loader = None
#
#
#
#     def ok(self):

# ----------------------------------------------------------------------------------------------------------------------
model = torch.load(r"")
valid_loader = r""
# ----------------------------------------------------------------------------------------------------------------------


model.eval()

gt_boxes, gt_classes = [], []
pred_boxes, pred_classes, pred_scores = [], [], []
#
for val_epoch, data in enumerate(valid_loader):
    img_ids, batch_imgs, batch_boxes, batch_classes = data
    inputs = list(img.type(torch.FloatTensor).to(device) for img in batch_imgs)  # 将图片全部转为 tensor

    # 不跟新参数，进行模型预测
    with torch.no_grad():
        pred = model(inputs)

    # 遍历每一个图片的检测结果
    for i in range(len(pred)):
        #
        pred_boxes.append(pred[i]["boxes"].cpu())
        pred_classes.append(pred[i]["labels"].cpu())
        pred_scores.append(pred[i]["scores"].cpu())
        #
        gt_boxes.append(batch_boxes[i])
        gt_classes.append(batch_classes[i])

    del inputs, pred

    # total_val_epochs = math.ceil(len(valid_dataset) / config.BATCH_SIZE)  # 向上取整
    # print("{} / {}".format(val_epoch + 1, total_val_epochs), end='\r')













