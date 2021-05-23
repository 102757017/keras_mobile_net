Batch Normalization是个啥

简单来说，BN将每一层的输入减去其在Batch中的均值，除以它的标准差，得到标准化的输入，加了BN之后Loss下降更快，最后能达到的效果也更好。

`BatchNormalization` 包含 2 个会在训练过程中更新的不可训练权重。它们是跟踪输入的平均值和方差的变量。

Keras用learning_phase机制来告诉模型当前的所处的模式

learning*_*phase设为1，表示训练模式，会自动将BN层的training参数设定为True

learning_phase设为0，表示测试模式，会自动将BN层的training参数设定为False



![image-20210523161708097](https://gitee.com/sunny_ho/image_bed/raw/master/img/20210523161718.png)



对于tensorflow 2.*版本，仅设定base_model.trainable = True，迁移学习时会导致训练时用了新数据集的均值和方差去做归一化，测试时用了imagenet数据集的移动均值和方差去做归一化，导致训练精度很高，测试精度很低。（与过拟合的现状很像）

原因是对于tensorflow 2.*版本，BN层的training参数默认为None，必须设定为True，BN层的移动均值和方差才能更新，

```
base_model = tf.keras.applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights='imagenet')

inputs = Input(shape=(32,32,3))
x=base_model(inputs,training=True)
base_model.trainable = True
x=GlobalAveragePooling2D()(x)
predictions=Dense(10,activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
```



我们通常希望训练和测试时网络中的配置一致，但冻结BN层的情况下，训练时用了新数据集的均值和方差去做归一化，测试时用了旧数据集的移动均值和方差去做归一化，导致精度下降。

解决方法是训练和测试时使用相同的移动均值和方差去做归一化。



## 解决方案：

打补丁，注意该补丁是打在独立版本的keras上的，TF自带的keras无法打补丁。

该补丁最高只支持keras2.2.4，keras2.2.4后端最高只支持到Tensorflow 1.13

!pip install -U --force-reinstall --no-dependencies git+https://github.com/datumbox/keras@fork/keras2.2.4



对于tensorflow 2.*版本，training默认为None，必须设定为True，BN层的移动均值和方差才能更新，仅设定base_model.trainable = True是无效的。

```
base_model = tf.keras.applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights='imagenet')

inputs = Input(shape=(32,32,3))
x=base_model(inputs,training=True)
base_model.trainable = True
x=GlobalAveragePooling2D()(x)
predictions=Dense(10,activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
```


