from torchvision.ops import misc
import torch

model = torch.load("/home/ldq/tj_v1.0.0.0/models/tj_classify_54_27756.pth")
for name, layer in model.named_modules():
    if isinstance(layer, misc.FrozenBatchNorm2d):
        layer.eps = 0.
        torch.save(model, "/home/ldq/tj_v1.0.0.0/models/tj_classify_54_27756_py36.pth")
