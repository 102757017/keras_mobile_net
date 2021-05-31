#需要手动更新tensorflow-estimator==2.3.0，否则导入tensorflow_hub时会出错
import tensorflow_hub as hub
import numpy as np
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical,Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint #训练途中自动保存模型
from imgaug import augmenters as iaa


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#训练整个数据集内存不够,因此定义一个生成器
class generator(Sequence):
    #数据增强
    def aug(self,x):
        #数据增强
        seq = iaa.Sequential([
            #iaa.RandAugment(n=3, m=7),
            iaa.Resize({"height": 96, "width": 96})
            ])
        x=seq.augment_images(x)
        #归一化,keras.applications中预训练模型是将数据归一化到[-1,1]之间，TF hub中的预训练模型是将数据归一化到[0,1]之间
        x=x.astype('float32')/255
        return x

    def __init__(self, x_set, y_set, batch_size):
        self.x_set , self.y_set = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y1 = self.y_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x1=self.aug(batch_x1)

        return batch_x1,batch_y1
  
      #该函数将在训练时每一个epoch结束的时候自动执行
    def on_epoch_end(self):
        print("epoch 结束")


#mnist需要转换为3通道图片
x_train=np.stack((x_train,)*3, axis=-1)
x_test=np.stack((x_test,)*3, axis=-1)


#独热编码
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


#https://...../feature_vector/5是去掉头部的“特征提取器”模型
#https://...../classification/5是完整的分类模型
#TF hub中的预训练模型不能更改inputshape，input_shape必须与下载url中的shape一致，否则会报错。
model_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5"
base_model=hub.KerasLayer(model_url,input_shape=(96,96,3))
base_model.trainable=True


inputs = Input(shape=(96,96,3))
x=base_model(inputs)
#将新头初始化为全零很重要
predictions=Dense(10,kernel_initializer='zeros',activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
base_model.trainable=True


# 编译模型
model.compile(loss='categorical_crossentropy',optimizer="Nadam",metrics=['accuracy'])



filepath = "weights-hub.hdf5"
# 每个epoch确认确认monitor的值，如果训练效果提升, 则将权重保存, 每提升一次, 保存一次
#mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode='auto')

#实现断点继续训练
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

batchsize=128
g=generator(x_train,y_train,batchsize)

#history =model.fit(x_train,y_train,batch_size=128,epochs=1,verbose=1,validation_split=0.1,callbacks=[checkpoint])
history =model.fit_generator(g, 
                    steps_per_epoch=int(x_train.shape[0]/batchsize),
                    epochs=1, 
                    verbose=1, 
                    callbacks=[checkpoint], 
                    validation_data=None, 
                    validation_steps=None, 
                    class_weight=None, 
                    max_queue_size=10, 
                    workers=2, 
                    use_multiprocessing=False, 
                    shuffle=True, 
                    #initial_epoch=0
                    )


# 评估模型
#model.evaluate返回的是一个list,其中第一个元素为loss指标，其它元素为metrias中定义的指标，metrias指定了N个指标则返回N个元素
loss,accuracy = model.evaluate(x_test,y_test,batch_size=1000)
print('\ntest loss',loss)
print('accuracy',accuracy)