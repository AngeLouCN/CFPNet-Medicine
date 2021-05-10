# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:09:44 2021

@author: angelou
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras import initializers
from keras.layers import SpatialDropout2D,Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,AveragePooling2D, UpSampling2D, BatchNormalization, Activation, add,Dropout,Permute,ZeroPadding2D,Add, Reshape
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU, ReLU, PReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import applications, optimizers, callbacks
import matplotlib
import keras
import tensorflow as tf
from keras.layers import *

# def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
#     '''
#     2D Convolutional layers
    
#     Arguments:
#         x {keras layer} -- input layer 
#         filters {int} -- number of filters
#         num_row {int} -- number of rows in filters
#         num_col {int} -- number of columns in filters
    
#     Keyword Arguments:
#         padding {str} -- mode of padding (default: {'same'})
#         strides {tuple} -- stride of convolution operation (default: {(1, 1)})
#         activation {str} -- activation function (default: {'relu'})
#         name {str} -- name of the layer (default: {None})
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
#     x = BatchNormalization(axis=3, scale=False)(x)

#     if(activation == None):
#         return x

#     x = Activation(activation, name=name)(x)

#     return x


# def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
#     '''
#     2D Transposed Convolutional layers
    
#     Arguments:
#         x {keras layer} -- input layer 
#         filters {int} -- number of filters
#         num_row {int} -- number of rows in filters
#         num_col {int} -- number of columns in filters
    
#     Keyword Arguments:
#         padding {str} -- mode of padding (default: {'same'})
#         strides {tuple} -- stride of convolution operation (default: {(2, 2)})
#         name {str} -- name of the layer (default: {None})
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
#     x = BatchNormalization(axis=3, scale=False)(x)
    
#     return x


def DCBlock(U, inp, alpha = 1.67):
    '''
    DC Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    #shortcut = inp

    #shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
   #                      int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3_1 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5_1 = conv2d_bn(conv3x3_1, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7_1 = conv2d_bn(conv5x5_1, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out1 = concatenate([conv3x3_1, conv5x5_1, conv7x7_1], axis=3)
    out1 = BatchNormalization(axis=3)(out1)
    
    conv3x3_2 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5_2 = conv2d_bn(conv3x3_2, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7_2 = conv2d_bn(conv5x5_2, int(W*0.5), 3, 3,
                        activation='relu', padding='same')
    out2 = concatenate([conv3x3_2, conv5x5_2, conv7x7_2], axis=3)
    out2 = BatchNormalization(axis=3)(out2)

    out = add([out1, out2])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out

# def ResPath(filters, length, inp):
#     '''
#     ResPath
    
#     Arguments:
#         filters {int} -- [description]
#         length {int} -- length of ResPath
#         inp {keras layer} -- input layer 
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     shortcut = inp
#     shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                          activation=None, padding='same')

#     out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

#     out = add([shortcut, out])
#     out = Activation('relu')(out)
#     out = BatchNormalization(axis=3)(out)

#     for i in range(length-1):

#         shortcut = out
#         shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                              activation=None, padding='same')

#         out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

#         out = add([shortcut, out])
#         out = Activation('relu')(out)
#         out = BatchNormalization(axis=3)(out)

#     return out

def DCUNet(height, width, channels):
    '''
    DC-UNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''

    inputs = Input((height, width, channels))

    dcblock1 = DCBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dcblock1)
    dcblock1 = ResPath(32, 4, dcblock1)

    dcblock2 = DCBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dcblock2)
    dcblock2 = ResPath(32*2, 3, dcblock2)

    dcblock3 = DCBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dcblock3)
    dcblock3 = ResPath(32*4, 2, dcblock3)

    dcblock4 = DCBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dcblock4)
    dcblock4 = ResPath(32*8, 1, dcblock4)

    dcblock5 = DCBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(dcblock5), dcblock4], axis=3)
    dcblock6 = DCBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(dcblock6), dcblock3], axis=3)
    dcblock7 = DCBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(dcblock7), dcblock2], axis=3)
    dcblock8 = DCBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(dcblock8), dcblock1], axis=3)
    dcblock9 = DCBlock(32, up9)

    conv10 = conv2d_bn(dcblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model


def conv2d_bn(x, filters, ksize, d_rate, strides,padding='same', activation='relu', groups=1, name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate = d_rate, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def CFPModule(inp, filters, d_size):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    x_inp = conv2d_bn(inp, filters//4, ksize=1, d_rate=1, strides=1)
    
    x_1_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
    x_1_2 = conv2d_bn(x_1_1, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
    x_1_3 = conv2d_bn(x_1_2, filters//8, ksize=3, d_rate=1, strides=1,groups=filters//8)
    
    x_2_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
    x_2_2 = conv2d_bn(x_2_1, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
    x_2_3 = conv2d_bn(x_2_2, filters//8, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//8)

    x_3_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
    x_3_2 = conv2d_bn(x_3_1, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
    x_3_3 = conv2d_bn(x_3_2, filters//8, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//8)
    
    x_4_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
    x_4_2 = conv2d_bn(x_4_1, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
    x_4_3 = conv2d_bn(x_4_2, filters//8, ksize=3, d_rate=d_size+1, strides=1, groups=filters//8)
    
    o_1 = concatenate([x_1_1,x_1_2,x_1_3], axis=3)
    o_2 = concatenate([x_2_1,x_2_2,x_2_3], axis=3)
    o_3 = concatenate([x_1_1,x_3_2,x_3_3], axis=3)
    o_4 = concatenate([x_1_1,x_4_2,x_4_3], axis=3)
    
    o_1 = BatchNormalization(axis=3)(o_1)
    o_2 = BatchNormalization(axis=3)(o_2)
    o_3 = BatchNormalization(axis=3)(o_3)
    o_4 = BatchNormalization(axis=3)(o_4)
    
    ad1 = o_1
    ad2 = add([ad1,o_2])
    ad3 = add([ad2,o_3])
    ad4 = add([ad3,o_4])
    output = concatenate([ad1,ad2,ad3,ad4],axis=3)
    #output = add([ad1,ad2,ad3,ad4])
    output = BatchNormalization(axis=3)(output)
    #output = Activation('relu')(output)
    output = conv2d_bn(output, filters, ksize=1, d_rate=1, strides=1,padding='valid')
    output = add([output, inp])

    return output


def CFPNet(height, width, channels):


    inputs = Input(shape=(height, width, channels))
    
    conv1=conv2d_bn(inputs, 32, 3, 1, 2)
    conv2 = conv2d_bn(conv1, 32, 3, 1, 1)
    conv3 = conv2d_bn(conv2, 32, 3, 1, 1)
    
    injection_1 = AveragePooling2D()(inputs)
    injection_1 = BatchNormalization(axis=3)(injection_1)
    injection_1 = Activation('relu')(injection_1)
    opt_cat_1 = concatenate([conv3,injection_1], axis=3)
    
    #CFP block 1
    opt_cat_1_0 = conv2d_bn(opt_cat_1, 64, 3, 1, 2)
    cfp_1 = CFPModule(opt_cat_1_0, 64, 2)
    cfp_2 = CFPModule(cfp_1, 64, 2)
    
    injection_2 = AveragePooling2D()(injection_1)
    injection_2 = BatchNormalization(axis=3)(injection_2)
    injection_2 = Activation('relu')(injection_2)
    opt_cat_2 = concatenate([cfp_2,opt_cat_1_0,injection_2], axis=3)
    
    #CFP block 2
    opt_cat_2_0 = conv2d_bn(opt_cat_2, 128, 3, 1, 2)
    cfp_3 = CFPModule(opt_cat_2_0, 128, 4)
    cfp_4 = CFPModule(cfp_3, 128, 4)
    cfp_5 = CFPModule(cfp_4, 128, 8)
    cfp_6 = CFPModule(cfp_5, 128, 8)
    cfp_7 = CFPModule(cfp_6, 128, 16)
    cfp_8 = CFPModule(cfp_7, 128, 16)
    
    injection_3 = AveragePooling2D()(injection_2)
    injection_3 = BatchNormalization(axis=3)(injection_3)
    injection_3 = Activation('relu')(injection_3)
    opt_cat_3 = concatenate([cfp_8,opt_cat_2_0,injection_3], axis=3)
    
    
    conv4 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same',activation='relu')(opt_cat_3)
    up_1 = concatenate([conv4,opt_cat_2])
    
    conv5 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same',activation='relu')(up_1)
    up_2 = concatenate([conv5, opt_cat_1],axis=3)    
    
    conv6 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same',activation='relu')(up_2)
    
    conv7 = conv2d_bn(conv6, 1, 1, 1, 1, activation='sigmoid', padding='valid')

    # output = UpSampling2D(size=(8,8),interpolation='bilinear')(conv7)
    
    model = Model(inputs=inputs, outputs=conv7)
    
    return model

dropout = 0.2
def unet(height,width,channels):
    inputs = Input(shape = (height, width, channels))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    return model

def ICNet(width, height, channels, n_classes=1):
    inp = Input(shape=(height, width, channels))
    x = Lambda(lambda x: x/1.0)(inp)

    # (1/2)
    y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='data_sub2')(x)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
    y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
    y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
    y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
    y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
    y = BatchNormalization(name='conv1_3_3x3_bn')(y)
    y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)
    
    y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)
    y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(name='padding1')(y_)
    y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
    y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
    y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
    y = Add(name='conv2_1')([y,y_])
    y_ = Activation('relu', name='conv2_1/relu')(y)

    y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding2')(y)
    y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
    y = BatchNormalization(name='conv2_2_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
    y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
    y = Add(name='conv2_2')([y,y_])
    y_ = Activation('relu', name='conv2_2/relu')(y)

    y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding3')(y)
    y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
    y = BatchNormalization(name='conv2_3_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
    y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
    y = Add(name='conv2_3')([y,y_])
    y_ = Activation('relu', name='conv2_3/relu')(y)

    y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
    y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_) 
    y_ = ZeroPadding2D(name='padding4')(y_)
    y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
    y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
    y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
    y = Add(name='conv3_1')([y,y_])
    z = Activation('relu', name='conv3_1/relu')(y)

    # (1/4)
    y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='conv3_1_sub4')(z)
    y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding5')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
    y = BatchNormalization(name='conv3_2_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
    y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
    y = Add(name='conv3_2')([y,y_])
    y_ = Activation('relu', name='conv3_2/relu')(y)

    y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding6')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
    y = BatchNormalization(name='conv3_3_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
    y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
    y = Add(name='conv3_3')([y,y_])
    y_ = Activation('relu', name='conv3_3/relu')(y)

    y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding7')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
    y = BatchNormalization(name='conv3_4_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
    y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
    y = Add(name='conv3_4')([y,y_])
    y_ = Activation('relu', name='conv3_4/relu')(y)

    y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
    y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
    y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
    y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
    y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
    y = Add(name='conv4_1')([y,y_])
    y_ = Activation('relu', name='conv4_1/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding9')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
    y = BatchNormalization(name='conv4_2_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
    y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
    y = Add(name='conv4_2')([y,y_])
    y_ = Activation('relu', name='conv4_2/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding10')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
    y = BatchNormalization(name='conv4_3_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
    y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
    y = Add(name='conv4_3')([y,y_])
    y_ = Activation('relu', name='conv4_3/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding11')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
    y = BatchNormalization(name='conv4_4_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
    y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
    y = Add(name='conv4_4')([y,y_])
    y_ = Activation('relu', name='conv4_4/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding12')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
    y = BatchNormalization(name='conv4_5_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
    y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
    y = Add(name='conv4_5')([y,y_])
    y_ = Activation('relu', name='conv4_5/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding13')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
    y = BatchNormalization(name='conv4_6_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
    y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
    y = Add(name='conv4_6')([y,y_])
    y = Activation('relu', name='conv4_6/relu')(y)

    y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
    y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
    y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
    y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding14')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
    y = BatchNormalization(name='conv5_1_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
    y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
    y = Add(name='conv5_1')([y,y_])
    y_ = Activation('relu', name='conv5_1/relu')(y)

    y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding15')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
    y = BatchNormalization(name='conv5_2_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
    y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
    y = Add(name='conv5_2')([y,y_])
    y_ = Activation('relu', name='conv5_2/relu')(y)

    y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding16')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
    y = BatchNormalization(name='conv5_3_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
    y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
    y = Add(name='conv5_3')([y,y_])
    y = Activation('relu', name='conv5_3/relu')(y)

    h, w = y.shape[1:3].as_list()
    pool1 = AveragePooling2D(pool_size=(h,w), strides=(h,w), name='conv5_3_pool1')(y)
    pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h,w)), name='conv5_3_pool1_interp')(pool1)
    pool2 = AveragePooling2D(pool_size=(h/2,w/2), strides=(h//2,w//2), name='conv5_3_pool2')(y)
    pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h,w)), name='conv5_3_pool2_interp')(pool2)
    pool3 = AveragePooling2D(pool_size=(h/3,w/3), strides=(h//3,w//3), name='conv5_3_pool3')(y)
    pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h,w)), name='conv5_3_pool3_interp')(pool3)
    pool6 = AveragePooling2D(pool_size=(h/4,w/4), strides=(h//4,w//4), name='conv5_3_pool6')(y)
    pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h,w)), name='conv5_3_pool6_interp')(pool6)

    y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
    y = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
    y = BatchNormalization(name='conv5_4_k1_bn')(y)
    aux_1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='conv5_4_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
    y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
    y = BatchNormalization(name='conv_sub4_bn')(y)
    y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
    y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)
    y = Add(name='sub24_sum')([y,y_])
    y = Activation('relu', name='sub24_sum/relu')(y)

    aux_2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub24_sum_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
    y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
    y_ = BatchNormalization(name='conv_sub2_bn')(y_)

    # (1)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_sub1')(x)
    y = BatchNormalization(name='conv1_sub1_bn')(y)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv2_sub1')(y)
    y = BatchNormalization(name='conv2_sub1_bn')(y)
    y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv3_sub1')(y)
    y = BatchNormalization(name='conv3_sub1_bn')(y)
    y = Conv2D(128, 1, name='conv3_sub1_proj')(y)
    y = BatchNormalization(name='conv3_sub1_proj_bn')(y)

    y = Add(name='sub12_sum')([y,y_])
    y = Activation('relu', name='sub12_sum/relu')(y)
    y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub12_sum_interp')(y)
    y = UpSampling2D(size=(4,4))(y)
    out = Conv2D(n_classes, 1, activation='sigmoid', name='conv6_cls')(y)


    model = Model(inputs=inp, outputs=out)

    return model


def initial_block(tensor):

    conv = Conv2D(filters=13, kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_conv', kernel_initializer='he_normal')(tensor)

    pool = MaxPooling2D(pool_size=(2, 2), name='initial_block_pool')(tensor)

    concat = concatenate([conv, pool], axis=-1, name='initial_block_concat')

    return concat

def bottleneck_encoder(tensor, nfilters, downsampling=False, dilated=False, asymmetric=False, normal=False, drate=0.1, name=''):

    y = tensor

    skip = tensor

    stride = 1

    ksize = 1

    if downsampling:

        stride = 2

        ksize = 2

        skip = MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{name}')(skip)

        skip = Permute((1,3,2), name=f'permute_1_{name}')(skip)       #(B, H, W, C) -> (B, H, C, W)

        ch_pad = nfilters - K.int_shape(tensor)[-1]

        skip = ZeroPadding2D(padding=((0,0),(0,ch_pad)), name=f'zeropadding_{name}')(skip)

        skip = Permute((1,3,2), name=f'permute_2_{name}')(skip)       #(B, H, C, W) -> (B, H, W, C)        

    

    y = Conv2D(filters=nfilters//4, kernel_size=(ksize, ksize), kernel_initializer='he_normal', strides=(stride, stride), padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)

    y = ReLU(name=f'prelu_1x1_{name}')(y)

    

    if normal:

        y = Conv2D(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', name=f'3x3_conv_{name}')(y)

    elif asymmetric:

        y = Conv2D(filters=nfilters//4, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same', use_bias=False, name=f'5x1_conv_{name}')(y)

        y = Conv2D(filters=nfilters//4, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same', name=f'1x5_conv_{name}')(y)

    elif dilated:

        y = Conv2D(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', dilation_rate=(dilated, dilated), padding='same', name=f'dilated_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)

    y = ReLU(name=f'prelu_{name}')(y)

    

    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False, name=f'final_1x1_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)

    y = SpatialDropout2D(rate=drate, name=f'spatial_dropout_final_{name}')(y)

    

    y = Add(name=f'add_{name}')([y, skip])

    y = ReLU(name=f'prelu_out_{name}')(y)

    

    return y

def bottleneck_decoder(tensor, nfilters, upsampling=False, normal=False, name=''):

    y = tensor

    skip = tensor

    if upsampling:

        skip = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1), padding='same', use_bias=False, name=f'1x1_conv_skip_{name}')(skip)

        skip = UpSampling2D(size=(2, 2), name=f'upsample_skip_{name}')(skip)

    

    y = Conv2D(filters=nfilters//4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1), padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)

    y = ReLU(name=f'prelu_1x1_{name}')(y)

    

    if upsampling:

        y = Conv2DTranspose(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', name=f'3x3_deconv_{name}')(y)

    elif normal:

        Conv2D(filters=nfilters//4, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same', name=f'3x3_conv_{name}')(y)    

    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)

    y = ReLU(name=f'prelu_{name}')(y)

    

    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False, name=f'final_1x1_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)



    y = Add(name=f'add_{name}')([y, skip])

    y = ReLU(name=f'relu_out_{name}')(y)

    

    return y

def ENet(height, width, channels,nclasses=1):

    print('. . . . .Building ENet. . . . .')

    img_input = Input(shape=(height, width, channels), name='image_input')



    x = initial_block(img_input)



    x = bottleneck_encoder(x, 64, downsampling=True, normal=True, name='1.0', drate=0.01)

    for _ in range(1,5):

        x = bottleneck_encoder(x, 64, normal=True, name=f'1.{_}', drate=0.01)



    x = bottleneck_encoder(x, 128, downsampling=True, normal=True, name=f'2.0')

    x = bottleneck_encoder(x, 128, normal=True, name=f'2.1')

    x = bottleneck_encoder(x, 128, dilated=2, name=f'2.2')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.3')

    x = bottleneck_encoder(x, 128, dilated=4, name=f'2.4')

    x = bottleneck_encoder(x, 128, normal=True, name=f'2.5')

    x = bottleneck_encoder(x, 128, dilated=8, name=f'2.6')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.7')

    x = bottleneck_encoder(x, 128, dilated=16, name=f'2.8')



    x = bottleneck_encoder(x, 128, normal=True, name=f'3.0')

    x = bottleneck_encoder(x, 128, dilated=2, name=f'3.1')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.2')

    x = bottleneck_encoder(x, 128, dilated=4, name=f'3.3')

    x = bottleneck_encoder(x, 128, normal=True, name=f'3.4')

    x = bottleneck_encoder(x, 128, dilated=8, name=f'3.5')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.6')

    x = bottleneck_encoder(x, 128, dilated=16, name=f'3.7')



    x = bottleneck_decoder(x, 64, upsampling=True, name='4.0')

    x = bottleneck_decoder(x, 64, normal=True, name='4.1')

    x = bottleneck_decoder(x, 64, normal=True, name='4.2')



    x = bottleneck_decoder(x, 16, upsampling=True, name='5.0')

    x = bottleneck_decoder(x, 16, normal=True, name='5.1')



    img_output = Conv2DTranspose(nclasses, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same', name='image_output')(x)

    img_output = Activation('sigmoid')(img_output)

    

    model = Model(inputs=img_input, outputs=img_output, name='ENET')

    print('. . . . .Build Compeleted. . . . .')

    return model


def conv_layer(inp,number_of_filters, kernel, stride):
    
    network = Conv2D(filters=number_of_filters, kernel_size=kernel, 
                      strides=stride, padding = 'same', activation='relu',
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def dilation_conv_layer(inp, number_of_filters, kernel, stride, dilation_rate):
    
    network = Conv2D(filters=number_of_filters, kernel_size=kernel, activation='relu',
                      strides=stride, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def BN_Relu(out):
    
    bacth_conv = BatchNormalization(axis=3)(out)
    relu_batch_norm = ReLU()(bacth_conv)
    return relu_batch_norm

def conv_one_cross_one(inp, number_of_classes):
    
    network = Conv2D(filters=number_of_classes, kernel_size=1, 
                      strides=1, padding = 'valid', activation='relu',
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def esp(inp, n_out):
    
    number_of_branches = 5
    n = int(n_out/number_of_branches)
    n1 = n_out - (number_of_branches - 1) * n
    
    # Reduce
    output1 = conv_layer(inp, number_of_filters=n, kernel=3, stride=2)
    
    # Split and Transform
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=3, stride=1, dilation_rate=1)
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=2)
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=4)
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=8)
    d16 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=16)
    
    add1 = dilated_conv2
    add2 = add([add1,dilated_conv4])
    add3 = add([add2,dilated_conv8])
    add4 = add([add3,d16])
    
    # Merge
    concat = concatenate([dilated_conv1,add1,add2,add3,add4], axis=3)
    concat = BN_Relu(concat)
    return concat

def esp_alpha(inp,n_out):
    number_of_branches = 5
    if n_out == 2:
        n = n1 = 2
    else:
        n = int(n_out/number_of_branches)
        n1 = n_out - (number_of_branches - 1) * n
    
    # Reduce
    output1 = conv_layer(inp, number_of_filters=n, kernel=3, stride=1)
    
    # Split and Transform
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=3, stride=1, dilation_rate=1)
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=2)
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=4)
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=8)
    dilated_conv16 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=16)
    
    add1 = dilated_conv2
    add2 = add([add1,dilated_conv4])
    add3 = add([add2,dilated_conv8])
    add4 = add([add3,dilated_conv16])
    
    # Merge
    concat = concatenate([dilated_conv1,add1,add2,add3,add4], axis=3)
    concat = BN_Relu(concat)
    return concat

def espnet(height,width,channels):
    inputs = Input(shape=(height,width,channels))
    conv_output = conv_layer(inputs, number_of_filters=16, kernel=3, stride=2)
    relu_ = BN_Relu(conv_output)
    
    avg_pooling1 = conv_output
    avg_pooling2 = AveragePooling2D()(avg_pooling1)
    avg_pooling2 = BN_Relu(avg_pooling2)
    
    concat1 = concatenate([avg_pooling1,relu_], axis=3)
    concat1 = BN_Relu(concat1)
    esp_1 = esp(concat1,64)
    esp_1 = BN_Relu(esp_1)

    esp_alpha_1 = esp_1
    
    esp_alpha_1 = esp_alpha(esp_alpha_1, 64)
    esp_alpha_1 = esp_alpha(esp_alpha_1, 64)
    concat2 = concatenate([esp_alpha_1,esp_1,avg_pooling2], axis=3)
    
    esp_2 = esp(concat2,128)
    esp_alpha_2 = esp_2

    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)

    
    concat3 = concatenate([esp_alpha_2,esp_2],axis = 3)
    pred = conv_one_cross_one(concat3, 16)
    
    deconv1 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='relu')(pred)
    conv_1 = conv_one_cross_one(concat2, 16)
    concat4 = concatenate([deconv1,conv_1], axis=3)
    esp_3 = esp_alpha(concat4, 16)
    deconv2 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='relu')(esp_3)
    conv_2 = conv_one_cross_one(concat1, 16)
    concat5 = concatenate([deconv2,conv_2], axis=3)
    conv_3 = conv_one_cross_one(concat5, 16)
    deconv3 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='relu')(conv_3)
    deconv3 = conv_one_cross_one(deconv3, 1)
    deconv3 = Activation('sigmoid')(deconv3)
    
    model = Model(inputs=inputs,outputs = deconv3)
    return model

# def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation=None,dilation=(1,1), name=None):
#     '''
#     2D Convolutional layers
    
#     Arguments:
#         x {keras layer} -- input layer 
#         filters {int} -- number of filters
#         num_row {int} -- number of rows in filters
#         num_col {int} -- number of columns in filters
    
#     Keyword Arguments:
#         padding {str} -- mode of padding (default: {'same'})
#         strides {tuple} -- stride of convolution operation (default: {(1, 1)})
#         activation {str} -- activation function (default: {'relu'})
#         name {str} -- name of the layer (default: {None})
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     x = Conv2D(filters, (num_row, num_col), strides=strides, dilation_rate=dilation, padding=padding, use_bias=False)(x)
#     x = BatchNormalization(axis=3, scale=False)(x)

#     # if(activation == None):
#     #     return x

#     x = Activation(activation, name=name)(x)

#     return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                        int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out




def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                          activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                              activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, channels))


    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=inputs, outputs=conv10)
    
    return model