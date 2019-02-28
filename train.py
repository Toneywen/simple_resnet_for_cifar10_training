#-*- coding: utf-8 -*-

# simple resnet network for cifar10 training 
# author: wenxiangyu
# time: 2019/02/28

import os
import numpy as np
import tensorflow as tf
import keras
import copy
import shutil
import utils
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

TOTAL_BATCH_SIZE = 128
EPOCHS = 120
INPUT_SHAPE = 32

def get_cifar10_statics():
    # xtrain, ytrain, xtest, ytest
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    # print 'len(x_train):'
    # print len(x_train)
    # trans to the 32bits float num
    x_train = np.float32(x_train)
    # print 'x_train:'
    # print x_train
    # make the pixel to be float lower than 1
    x_train *= 1. / 255
    # print 'x_train *= 1 / 255: '
    # print x_train
    # get the mean of the whole image
    mean = np.mean(x_train, axis=(0, 1, 2))
    # [0.491401   0.4821591  0.44653094]
    # print 'mean: '
    # print mean
    # standard deviation
    std = np.std(x_train, axis=(0, 1, 2))
    # [0.2470328  0.24348424 0.26158753]
    # print 'std: '
    # print std
    return np.float32(mean), np.float32(std)

def get_resnet_model(mean, std, weight_decay=1e-5, mask_dict=None, use_mask=False):
    def normalize(input, mean, std):
        return (input / 255. - tf.constant(mean)) / tf.constant(std)
    # 使用l2正则化
    kernel_regularizer = keras.regularizers.l2(weight_decay)

    input = keras.layers.Input((INPUT_SHAPE, INPUT_SHAPE, 3), name='input', dtype='uint8')
    normalized_input = keras.layers.Lambda(lambda input: tf.cast(input, 'float'), name='cast_dtype')(input)
    normalized_input = keras.layers.Lambda(lambda input: normalize(input, mean, std), name='normalized_intput')(normalized_input)

    l = keras.layers.Conv2D(64, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(normalized_input)

    layer = keras.layers.BatchNormalization(name='res2a_bn2a')(l)
    layer = keras.layers.Conv2D(64, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res2a_bn2b')(layer)
    layer = keras.layers.Conv2D(64, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)

    layer = keras.layers.BatchNormalization(name='res2a_bn2c')(layer)

    # res_2a shortcut
    l = keras.layers.BatchNormalization(name='res2a_bn1')(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(64, 1, use_bias=False, kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'))(l)
    l = keras.layers.BatchNormalization(name='res2a_bn2')(l)
    l = keras.layers.add([l,layer])  #直接相加
    # l1 = l+layer

    #res_2b
    layer = keras.layers.BatchNormalization(name='res2b_bn2a')(l)
    layer = keras.layers.Conv2D(64, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res2b_bn2b')(layer)
    layer = keras.layers.Conv2D(64, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res2b_bn2c')(layer)

    # res_2b shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.add([l, layer])
    l = keras.layers.MaxPool2D(2, name='res_2b_pool', strides=2, padding='same')(l)

    #res_3a
    layer = keras.layers.BatchNormalization(name='res3a_bn2a')(l)
    layer = keras.layers.Conv2D(128, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res3a_bn2b')(layer)
    layer = keras.layers.Conv2D(128, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res3a_bn_2c')(layer)

    # res_3a shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(128, 1, use_bias=False, kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.VarianceScaling(2., 'fan_out'))(l)
    l = keras.layers.BatchNormalization(name='res3a_bn')(l)
    l = keras.layers.add([l, layer])

    #res_3b
    layer = keras.layers.BatchNormalization(name='res3b_bn2a')(l)
    layer = keras.layers.Conv2D(128, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res3b_bn2b')(layer)
    layer = keras.layers.Conv2D(128, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res3b_bn2c')(layer)

    # res_3b shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.add([l, layer])
    l = keras.layers.MaxPool2D(2, name='res_3b_pool', strides=2, padding='same')(l)

    #res_4a
    layer = keras.layers.BatchNormalization(name='res4a_bn2a')(l)
    layer = keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res4a_bn2b')(layer)
    layer = keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res4a_bn2c')(layer)

    # res_4a shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(256, 1, use_bias=False, kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.VarianceScaling(2., 'fan_out'))(l)
    l = keras.layers.BatchNormalization(name='res4a_bn')(l)
    l = keras.layers.add([l, layer])

    #res_4b
    layer = keras.layers.BatchNormalization(name='res4b_bn2a')(l)
    layer = keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res4b_bn2b')(layer)
    layer = keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res4b_bn2c')(layer)

    # res_4b shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.add([l, layer])
    l = keras.layers.MaxPool2D(name='res_4b_pool', strides=2, padding='same')(l)

    # res_5a
    layer = keras.layers.BatchNormalization(name='res5a_bn2a')(l)
    layer = keras.layers.Conv2D(512, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res5a_bn2b')(layer)
    layer = keras.layers.Conv2D(512, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res5a_bn2c')(layer)

    # res_5a shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(512, 1, use_bias=False, kernel_regularizer=kernel_regularizer,
                            kernel_initializer=keras.initializers.VarianceScaling(2., 'fan_out'))(l)
    l = keras.layers.BatchNormalization(name='res5a_bn')(l)
    l = keras.layers.add([l, layer])

    #res_5b
    layer = keras.layers.BatchNormalization(name='res5b_bn2a')(l)
    layer = keras.layers.Conv2D(512, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res5b_bn2b')(layer)
    layer = keras.layers.Conv2D(512, 3, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(2., mode='fan_out'),
                            kernel_regularizer=kernel_regularizer)(layer)
    layer = keras.layers.BatchNormalization(name='res5b_bn2c')(layer)

    # res_5b shortcut
    l = keras.layers.ReLU()(l)
    l = keras.layers.add([l, layer])
    layer = keras.layers.ReLU()(l)

    layer = keras.layers.GlobalAveragePooling2D(name='gap')(layer)
    layer = keras.layers.Dropout(0.85, name='drop_fc')(layer)
    layer = keras.layers.Dense(10, kernel_initializer=keras.initializers.VarianceScaling(2.),
                               kernel_regularizer=kernel_regularizer)(layer)
    # layer = Flatten()(layer)
    softmax_output = keras.layers.Softmax()(layer)

    model = keras.models.Model(input, softmax_output)
    loss = [keras.losses.sparse_categorical_crossentropy]
    metrics = [keras.metrics.sparse_categorical_accuracy]
    model.compile(keras.optimizers.SGD(lr=0.), loss=loss, metrics=metrics)

    model.summary()
    return model

def train(model, load_data_func):
    # load datasets of cifar10
    (x_train, y_train), (x_val, y_val) = load_data_func()
    n_train_samples = len(x_train)
    eval_freq = 1
    # 用以生成一个batch的图像数据，支持实时数据提升
    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                           shear_range=0.1,
                                                           zoom_range=0.1,
                                                           width_shift_range=0.1,
                                                           height_shift_range=0.1,
                                                           channel_shift_range=0.1,
                                                           brightness_range=[0.85, 1.15],
                                                           horizontal_flip=True)
    # 模型名称
    model_name = os.path.basename(__file__).split('.')[0]
    # 保存地址
    log_dir = './trained_model/%s' %model_name

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # 模型保存的位置
    model.save(os.path.join(log_dir, 'model.h5'))

    # // 向下取整
    # 迭代的次数  总数 / batchsize
    steps_per_epoch = np.ceil(float(n_train_samples) / TOTAL_BATCH_SIZE) // eval_freq

    # 默认 EPOCHS
    epochs = EPOCHS

    # callback
    # 在one_epoch_end时模型数据写入文件
    cbks = [
        utils.LRTensorBoard('%s' %log_dir),
        keras.callbacks.ModelCheckpoint('%s/cp-{epoch:04d}.ckpt' %log_dir, save_weights_only=True),
        keras.callbacks.ModelCheckpoint('%s/best_cp.ckpt' %log_dir, save_best_only=True, save_weights_only=True)
    ]

    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=TOTAL_BATCH_SIZE),
                            steps_per_epoch=steps_per_epoch,
                            callbacks=cbks,
                            epochs=epochs, validation_data=(x_val, y_val))

if __name__ == '__main__':
    # Create a Keras Model
    mean, std = get_cifar10_statics()
    load_data_func = keras.datasets.cifar10.load_data

    model = get_resnet_model(mean, std, weight_decay=1e-5)

    plot_model(model,to_file='ResNet.png')

    train(model, load_data_func)