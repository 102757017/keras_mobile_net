from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 创建预训练模型
##include_top=False，去掉最后的3个全连接层，用来做fine-tuning专用，专门开源了这类模型。 
base_model = MobileNet(weights='imagenet', include_top=False)

# 增加全局平均池化层，全局平均池化用来代替全连接层，可以减少参数，同时减少过拟合。
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 增加全连接层
x = Dense(1024, activation='relu')(x)
# softmax激活函数用户分类
predictions = Dense(200, activation='softmax')(x)

# 预训练模型与新加层的组合
model = Model(inputs=base_model.input, outputs=predictions)

# 只训练新加的Top层，冻结MobileNet所有层
#也可以使用base_model.get_layer('layer_name').trainable = False
#定义层的时候可以指定name参数，如：x=Dense(784,activation='relu',name="my_lay")(x)
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 训练模型
model.fit_generator(...)

'''
# 第二种设置：冻结前249层，训练后249层
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 编译模型
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 训练模型
model.fit_generator(...)
'''
