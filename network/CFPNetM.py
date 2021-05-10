# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:37:33 2021

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
    CFP module for medicine
    
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
    output = BatchNormalization(axis=3)(output)
    output = conv2d_bn(output, filters, ksize=1, d_rate=1, strides=1,padding='valid')
    output = add([output, inp])

    return output

def CFPNetM(height, width, channels):

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
    
    model = Model(inputs=inputs, outputs=conv7)
    
    return model