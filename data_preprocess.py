'''
This is a data preprocess for ETHZ food 101
Developed in Python 3.5 and OpenCV
'''

BGR = [87.78324737,113.29841007,139.17742277] # mean of the whole food-101 dataset

import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# split train/test 800/200 per class
train_image = np.zeros((80800,224,224,3),dtype=np.uint8)
train_category = np.zeros((80800,1),dtype=np.uint8)
train_img_count = 0
test_image = np.zeros((20200,224,224,3),dtype=np.uint8)
test_category = np.zeros((20200,1),dtype=np.uint8)
test_img_count = 0
cat_count = 0

# load image
path = 'ETHZ-FOOD-101/food-101/images'
Classes = os.listdir(path)

for Class in Classes:
    train_category[cat_count*800:(cat_count+1)*800] = cat_count # assign class
    test_category[cat_count*200:(cat_count+1)*200] = cat_count  # assign class
    cat_count+=1

    Ids = os.listdir(path + '/' + Class)
    test_idx = np.random.choice(1000, 200, replace=False) # random sampling 200 idx as testing images

    for Id in Ids:
        img = cv2.imread(path+'/'+Class+'/'+Id)

        # grey world method
        # bgr = np.array(img).mean(axis=(0,1))
        out_bgr = np.mean(BGR)*img/BGR
        max_bgr = np.max(out_bgr)
        img_grey = np.uint8(out_bgr/max_bgr*255.)
        #img_grey = np.uint8(img/bgr*255)

        # histogram equalization
        img_yuv = cv2.cvtColor(img_grey, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to BGR format
        img_hist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # center crop
        if img.shape[0]>img.shape[1]:
            img_crop = img_hist[np.int(np.floor((img.shape[0] - img.shape[1])/2)):
                                np.int(np.ceil((img.shape[0] + img.shape[1])/2)),:,:]
        elif img.shape[0]<img.shape[1]:
            img_crop = img_hist[:,np.int(np.floor((img.shape[1] - img.shape[0])/2)):
                                np.int(np.ceil((img.shape[1] + img.shape[0])/2)),:]
        else:
            img_crop = img_hist
        # reshape
        img_output = cv2.resize(img_crop, (224, 224), interpolation = cv2.INTER_AREA)

        # save to np array
        if Ids.index(Id) in test_idx:
            test_image[test_img_count, :, :, :] = img_output
            test_img_count+=1
        else:
            train_image[train_img_count,:,:,:] = img_output
            train_img_count+=1
            # show 1 image for each class
            #if  train_img_count%800==0:
                #cv2.imshow('input image', img)
                #cv2.imshow('after grey world', img_grey)
                #cv2.imshow('after histogram', img_hist)
                #cv2.imshow('after center crop', img_crop)
                #cv2.imshow('ouput image', img_output)
                #cv2.waitKey(0)
                #plt.imshow(img_output)
                #plt.show()

# shuffle and save to h5
train_image, train_category = shuffle(train_image, train_category)
with h5py.File('food101_n80800_r224x224x3_train.h5', 'w') as hf:
    hf.create_dataset("image",  data=train_image)
    hf.create_dataset("category", data=train_category)

test_image, test_category = shuffle(test_image, test_category)
with h5py.File('food101_n20200_r224x224x3_test.h5', 'w') as hf:
    hf.create_dataset("image",  data=test_image)
    hf.create_dataset("category", data=test_category)