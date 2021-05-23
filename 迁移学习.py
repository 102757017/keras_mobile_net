from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#from tensorflow.keras.applications import MobileNetV3Large #TF2.0专用
#from tensorflow.keras.applications.efficientnet import EfficientNetB0  #TF2.0专用
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,Input,BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau #动态调整学习率
from tensorflow.keras.callbacks import ModelCheckpoint #训练途中自动保存模型
import tensorflow.keras.backend as K
import os
from imgaug import augmenters as iaa



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


#mnist需要转换为3通道图片
x_train=np.stack((x_train,)*3, axis=-1)
x_test=np.stack((x_test,)*3, axis=-1)


#数据集转换为32*32的分辨率
seq = iaa.Sequential([iaa.Resize({"height": 32, "width": 32})])
x_train=seq.augment_images(x_train)
x_test=seq.augment_images(x_test)



#归一化
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

#独热编码
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)




# 创建预训练模型
##include_top=False，去掉最后的3个全连接层，用来做fine-tuning专用，专门开源了这类模型。 
mobilenet = MobileNetV2(input_shape=(32,32,3),weights='imagenet',include_top=False)

'''
# 只训练新加的Top层，冻结MobileNet所有层
#也可以使用base_model.get_layer('layer_name').trainable = False
#定义层的时候可以指定name参数，如：x=Dense(784,activation='relu',name="my_lay")(x)
for layer in mobilenet.layers:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False
'''

'''
# 第二种设置：冻结前249层，训练后249层
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

'''

inputs = Input(shape=(32,32,3))

x=mobilenet(inputs,training=True)
#增加全局平均池化层，简称GAP,用GAP替代FC全连接层。有两个有点：一是GAP在特征图与最终的分类间转换更加简单自然；二是不像FC层需要大量训练调优的参数，降低了空间参数会使模型更加健壮，抗过拟合效果更佳。
#假设输入是h × w × d，h × w 会被平均化成一个值。
x=GlobalAveragePooling2D()(x)
predictions=Dense(10,activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)


# 编译模型
model.compile(loss='categorical_crossentropy',optimizer="Nadam",metrics=['accuracy'])


filepath = "weights-improvement.hdf5"
# 每个epoch确认确认monitor的值，如果训练效果提升, 则将权重保存, 每提升一次, 保存一次
#mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode='auto')

#实现断点继续训练
if os.path.exists(filepath):
    base_model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=3, mode='auto')


# 训练模型
history = model.fit(x_train,y_train,batch_size=128,epochs=1,verbose=1,callbacks=[reduce_lr,checkpoint],validation_split=0.1)


# 评估模型
#model.evaluate返回的是一个list,其中第一个元素为loss指标，其它元素为metrias中定义的指标，metrias指定了N个指标则返回N个元素
loss,accuracy = model.evaluate(x_test,y_test,batch_size=1000)
print('\ntest loss',loss)
print('accuracy',accuracy)
