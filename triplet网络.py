#!/usr/bin/python
# # -*- coding: UTF-8 -*-
from keras.preprocessing import image
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import Model #导入函数式模型
from keras.layers import Input #导入输入数据层

from keras.layers import Convolution2D #导入卷积层
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D #导入池化层
from keras.layers import Dense #导入全连接神经层
from keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元
from keras.layers import Activation #导入激活函数
import keras
#from keras.layers import K
from keras import backend as K


def triplet_loss(y_true, y_pred):
    """
    Triplet Loss的损失函数
    """

    anc, pos, neg = y_pred[:, 0:1024], y_pred[:, 1024:2048], y_pred[:, 2048:]

    # 欧式距离
    pos_dist = K.sum(K.square(anc - pos), axis=1, keepdims=True)
    neg_dist = K.sum(K.square(anc - neg), axis=1, keepdims=True)
    basic_loss = pos_dist - neg_dist + 0.2

    loss = K.maximum(basic_loss, 0.0)
    temp=K.greater(loss,0 )
    n=K.sum(K.cast(temp, dtype="float32"))+K.epsilon()
    loss=K.sum(loss)/n

    return loss


#include_top=True，完整的模型
#include_top=False，去掉最后的3个全连接层，用来做fine-tuning专用，专门开源了这类模型。
#迁移学习必须指定输入图片的shape，否则会默认为(224, 224)，图片的宽高必须大于197
base_model = MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False)



#函数式模型
inputs1 = Input(shape=(224, 224, 3))
inputs2 = Input(shape=(224, 224, 3))
inputs3 = Input(shape=(224, 224, 3))

# 增加全局平均池化层
x1=base_model(inputs1)
x2=base_model(inputs2)
x3=base_model(inputs3)


midputs = Input(shape=(7, 7, 1024))
p=Dropout(0.25)(midputs)
y = GlobalAveragePooling2D()(p)
y=Dense(128, activation='relu')(y)
#共享权重必须使用自定义模型,只创建一个Model对象；若向不共享权重，就分别创建两个对象。
model2=Model(inputs=midputs, outputs=y)

y1 = model2(x1)
y2 = model2(x2)
y3 = model2(x3)

#将两个输入并联
y = keras.layers.concatenate([y1, y2,y3], axis=1)

# softmax激活函数用户分类
#predictions = Dense(1, activation='softmax')(y)

# 预训练模型与新加层的组合
model = Model(inputs=[inputs1,inputs2,inputs3], outputs=y)

# 只训练新加的Top层，冻结MobileNet所有层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='rmsprop', loss=triplet_loss)

print(model.summary())

from keras.utils import plot_model
import pydot_ng as pydot
import os

#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"G:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"
plot_model(model, to_file='triplet模型结构.png',show_shapes=True)

# 训练模型
#model.fit_generator(...)
