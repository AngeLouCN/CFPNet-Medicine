# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:05:16 2021

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
from model.model import DCUNet,CFPNet, unet, ICNet, ENet, espnet,MultiResUnet
from keras.preprocessing.image import ImageDataGenerator
import time
from loss import dice_coef, jacard, dice_coef_loss, iou_loss, tversky, tversky_loss, focal_tversky, generalized_dice_coeff, generalized_dice_loss
from network.DCUNet import DCUNet
from network.CFPNetM import CFPNetM
from network.ICNet import ICNet
from network.ENet import ENet
from network.ESPNet import ESPNet
from network.MultiResUNet import MultiResUnet
from network.UNet import UNet
from keras.models import load_model
import segmentation_models as sm
# prepare training and testing set
X = []
Y = []

for i in range(612):
    path = 'D:\\CVC-ClinicDB\\Original\\'+ str(i+1)+'.tif'
    img = cv2.imread(path,1)
    resized_img = cv2.resize(img,(256, 192), interpolation = cv2.INTER_CUBIC)   
    X.append(resized_img)
    
for i in range(612):
    
    path2 = 'D:\\CVC-ClinicDB\\Ground Truth\\' + str(i+1)+'.tif'
    msk = cv2.imread(path2,0)
    resized_msk = cv2.resize(msk,(256, 192), interpolation = cv2.INTER_CUBIC) 
    Y.append(resized_msk)
    
# # ######################################################################
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

##### If gray-level image, use this code
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
# Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))
# X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
# X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
Y_train = Y_train.astype('float32') / 255
Y_test = Y_test.astype('float32') / 255

Y_train = np.round(Y_train,0)	
Y_test = np.round(Y_test,0)	

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)




def saveModel(model):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass
    
    fp = open('models/modelP.json','w')
    fp.write(model_json)
    model.save('models/modelW.h5')


def evaluateModel(model, X_test, Y_test, batchSize):
    
    try:
        os.makedirs('results')
    except:
        pass 
    
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    
    # for i in range(10):

    #     plt.figure(figsize=(20,10))
    #     plt.subplot(1,3,1)
    #     plt.imshow(X_test[i])
    #     plt.title('Input')
    #     plt.subplot(1,3,2)
    #     plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
    #     plt.title('Ground Truth')
    #     plt.subplot(1,3,3)
    #     plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
    #     plt.title('Prediction')

    #     intersection = yp[i].ravel() * Y_test[i].ravel()
    #     union = yp[i].ravel() + Y_test[i].ravel() - intersection

    #     jacard = (np.sum(intersection)/np.sum(union))  
    #     plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

    #     plt.savefig('results/'+str(i)+'.png',format='png')
    #     plt.close()
    
    jacard = 0
    dice = 0
    
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    

    fp = open('models/log.txt','a')
    fp.write(str(jacard)+'\n')
    fp.close()

    fp = open('models/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best.txt','w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)


def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):

    
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)     
        evaluateModel(model,X_test, Y_test,batchSize)

    return model 

###### model construct for (UNet MultiResUnet DCUNet CFPNetM ICNet ENET ESPNet)
model = CFPNetM(height=192, width=256, channels=3)

model.compile(optimizer='adam', loss=generalized_dice_loss, metrics=[dice_coef, jacard, 'accuracy'])
model.summary()
saveModel(model)

####### for use efficientnet_b0, inception_v3 and mobilenet_v2 as backbone, use thi code
# model = sm.Unet(backbone_name='efficientnetb0',  #  MobileNet v2 = 'mobilenetv2'
#                                                  #  Inception v3 = 'inceptionv3'
#                                                  #  EfficientNet_b0 = 'efficientnetb0'
#                 input_shape=(height,width, 3), classes=1)
# model.summary()
# model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
# saveModel(model)
##########################################################################################

fp = open('models/log.txt','w')
fp.close()
fp = open('models/best.txt','w')
fp.write('-1.0')
fp.close()
    
trainStep(model, X_train, Y_train, X_test, Y_test, epochs=1, batchSize=4)


#####################  print result
# y_result = model.predict(x=X_test, batch_size=10, verbose=1)
# y_result = np.round(y_result,0)   
# # path_result = 'D:/Brest SPIE/isic2018/fold1/result_icnet/'
# path_result = 'D:/CT/result_cfpnet/'
# # path_result = 'D:/Brest SPIE/DRIVE/exp5/result_icnet/'
# for i in range(14):
#     cv2.imwrite(path_result+'seg_'+str(i+1)+'.png',y_result[i]*255) #show predict results
#     cv2.imwrite(path_result+'img_'+str(i+1)+'.png',X_test[i]*255) #show predict results
#     cv2.imwrite(path_result+'org_'+str(i+1)+'.png',Y_test[i]*255) # show the label

######################### FLOPs 
# def get_flops(model):
#     run_meta = tf.RunMetadata()
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
 
#     # We use the Keras session graph in the call to the profiler.
#     flops = tf.profiler.profile(graph=K.get_session().graph,
#                                 run_meta=run_meta, cmd='op', options=opts)
 
#     return flops.total_float_ops  # Prints the "flops" of the model.
 
 
# # .... Define your model here ....
# print(get_flops(model))

