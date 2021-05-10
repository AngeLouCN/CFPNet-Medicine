# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:54:57 2021

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

def ESPNet(height,width,channels):
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