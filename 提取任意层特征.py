#!/usr/bin/python
# # -*- coding: UTF-8 -*-
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from keras.utils import plot_model
from matplotlib import pyplot as plt

#【0】MobileNet模型，加载预训练权重
base_model = MobileNet(weights='imagenet')
#print(base_model.summary()) 

#【1】创建一个新model, 使得它的输出(outputs)是 MobileNet 中任意层的输出(output)
#定义层的时候可以指定name参数，如：x=Dense(784,activation='relu',name="my_lay")(x)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('dropout').output)
print(model.summary())                                 # 打印模型概况

#【2】从网上下载一张图片，保存在当前路径下
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224)) # 加载图片并resize成224x224

#【3】将图片转化为4d tensor形式
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#【4】数据预处理
x = preprocess_input(x) #去均值中心化，preprocess_input函数详细功能见注释

#【5】提取特征
block4_pool_features = model.predict(x)
print(block4_pool_features.shape) #(1, 14, 14, 512)
