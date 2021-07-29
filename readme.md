
# 说明


### todo

* 测试一下写的几个训练代码是不是都能进行训练
    * vgg [√]
    * faster rcnn [√]
    * mask rcnn [×]

* 每一个训练代码都要传入训练集和验证集，直接在训练的时候验证
    * vgg [√]
    * faster rcnn [×]
    * mask rcnn [×]

* 保存最优的训练结果
    * vgg [√]
    * faster rcnn [√]
    * mask rcnn [×]

* 设定模型输入图片的最大值
    * vgg [√]
    * faster rcnn [√] 未测试
    * mask rcnn [×]

* 训练模型可以提供只跑验证集的模式 test_only，这样其实基本就用不着 test 代码了


### 元参考

* 尽量看懂并重写 torchvision 的官方参考代码 : https://github.com/pytorch/vision.git

* 学习官方的格式和思维方式，对官方的代码进行简化并注释

### 各种数据集格式整理

* COCO
* VOC

