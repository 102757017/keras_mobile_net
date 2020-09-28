#!/usr/bin/python
# # -*- coding: UTF-8 -*-
from keras.preprocessing import image
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


#include_top=True，完整的模型
#include_top=False，去掉最后的3个全连接层，用来做fine-tuning专用，专门开源了这类模型。 
model = MobileNetV2(weights='imagenet')
print(model.summary())



img_path = "elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))
#将输入数据转换为0~1之间
img = image.img_to_array(img) / 255.0
# 为batch添加第四维,axis=0表示在0位置添加,因为MobileNet的Iput层结构是（None,224,224,3）
img = np.expand_dims(img, axis=0)
print(img.shape)

predictions = model.predict(img)
print('Predicted:', decode_predictions(predictions, top=3)[0])
print(predictions)

description = decode_predictions(predictions, top=3)[0][0][1]

src = cv2.imread(img_path)
cv2.putText(src, description, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
cv2.imshow("Predicted", src)
cv2.waitKey()



from keras.utils import plot_model
import pydot_ng as pydot
import os

#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"G:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"
plot_model(model, to_file='模型结构.png',show_shapes=True)
