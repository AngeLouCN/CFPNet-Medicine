import os, glob
import cv2
import numpy as np
from keras.models import load_model
import segmentation_models as sm
import imageio


def saveModel(model):
    model_json = model.to_json()
    try:
        os.makedirs('results/fold/models')
    except:
        pass

    fp = open('results/fold/models/modelP.json', 'w')
    fp.write(model_json)
    model.save('results/fold/models/modelW.h5')

def evaluateModel(model, X_test, Y_test, batchSize):
    try:
        os.makedirs('results/fold/models')
    except:
        pass

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=2)


    yp = np.round(yp, 0)


    jacard = 0
    dice = 0

    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection) / np.sum(union))

        dice += (2. * np.sum(intersection)) / (np.sum(yp_2) + np.sum(y2))

    jacard /= len(Y_test)
    dice /= len(Y_test)

    print('Jacard Index : ' + str(jacard))
    print('Dice Coefficient : ' + str(dice))

    fp = open('results/fold/models/log.txt', 'a')
    fp.write(str(jacard) + '\n')
    fp.close()

    fp = open('results/fold/models/best.txt', 'r')
    best = fp.read()
    fp.close()

    if (jacard > float(best)):
        print('***********************************************')
        print('Jacard Index improved from ' + str(best) + ' to ' + str(jacard))
        print('***********************************************')
        fp = open('results/fold/models/best.txt', 'w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)

def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch + 1))

        model.fit(
            x=X_train,
            y=Y_train,
            batch_size=batchSize,
            epochs=1,
            verbose=2
        )

        evaluateModel(model, X_test, Y_test, batchSize)

    return model


### set parameters
# ISBI 256, 256
# DRIVE 256, 256
# Breast 256, 128
# CVC ISIC 256, 192

Train_num, Test_num = 2075, 519
img_width=256
img_height=192
batch_size = 15

### input data
X_train = []
Y_train= []
X_test = []
Y_test =[]


for i in range(Train_num):

    path1 = r'datasets\\isic\\fold\\train_img/org_'+ str(i+1) + '.png'
    img = cv2.imread(path1, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    X_train.append(resized_img)



for i in range(Train_num):
    path2 = r'datasets\\isic\\fold\\train_label/label_' + str(i+1) + '.png'
    msk = imageio.imread(path2)
    resized_msk = cv2.resize(np.array(msk).squeeze(), (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    Y_train.append(resized_msk)


for i in range(Test_num):

    path1 = r'datasets\\isic\\fold\\test_img/org_' + str(i+1) + '.png'
    img = cv2.imread(path1, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    X_test.append(resized_img)



for i in range(Test_num):
    path2 = r'datasets\\isic\\fold\\test_label/label_' + str(i+1) + '.png'
    msk = imageio.imread(path2)
    resized_msk = cv2.resize(np.array(msk).squeeze(), (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    Y_test.append(resized_msk)


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

Y_train = np.expand_dims(Y_train,axis=3)
Y_test = np.expand_dims(Y_test,axis=3)

X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

Y_train = np.round(Y_train, 0)
Y_test = np.round(Y_test, 0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



# fold for output
try:
    os.makedirs('results/fold')
except:
    pass


# construct the model
model = sm.Unet(backbone_name='efficientnetb0',  #  MobileNet v2 = 'mobilenetv2'
                                                 #  Inception v3 = 'inceptionv3'
                                                 #  EfficientNet_b0 = 'efficientnetb0'
                input_shape=(img_height,img_width, 3),
                classes=1)

model.summary()

# compile model
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# train model
saveModel(model)

fp = open('results/fold/models/log.txt', 'w')
fp.close()
fp = open('results/fold/models/best.txt', 'w')
fp.write('-1.0')
fp.close()

trainStep(model, X_train, Y_train, X_test, Y_test, epochs=400, batchSize=batch_size*2)

# print results of the best model
best_model = load_model('results/fold/models/modelW.h5',
                        custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
                                        'iou_score': sm.metrics.iou_score})
y_result = best_model.predict(x=X_test, batch_size=batch_size, verbose=2)
y_result = np.round(y_result, 0)

path_result = r'results/fold/'
for i in range(len(X_test)):  # output images
    cv2.imwrite(path_result + 'seg_' + str(i + 1) + '.png', y_result[i] * 255)  # show predict results
    cv2.imwrite(path_result + 'org_' + str(i + 1) + '.png', Y_test[i] * 255)  # show the label





