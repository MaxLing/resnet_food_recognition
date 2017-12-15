'''
This is a trial on dataset ETHZ food 101
This Neural Networks are developed in Python 3.5 and TensorFlow
'''

import numpy as np
import tensorflow as tf
import h5py

## generator for loading data
def generate_xy(path, batch_size):
    while 1:
        with h5py.File(path, 'r') as hf:
            for i in range(0, len(hf["category"]), batch_size):
                x = np.array(hf["image"][i:i+batch_size],dtype=np.uint8).reshape((-1,224,224,3))
                y = np.eye(101,dtype=np.uint8)[hf["category"][i:i+batch_size].reshape((-1,))]
                yield (x, y)

from keras.applications.resnet50 import ResNet50
from keras import optimizers, metrics, models
from keras.layers import Input, Flatten, Dense
from keras.models import Model, model_from_json
'''
# load model
model_resnet50 = models.load_model('model/resnet50_5_raw_rmspop.h5')
for layer in model_resnet50.layers[1].layers: # freeze or un-freeze resnet50
    layer.trainable = True
# with open("model/resnet50_full.json", "r") as json_file:
#    model_resnet50 = model_from_json(json_file.read())
# model_resnet50.load_weights("model/resnet50_5_raw_rmspop_full.h5")
'''
# model initialization (comment out after first save)
#Get back the convolutional part of a resnet network trained on ImageNet
model_resnet50_conv = ResNet50(weights='imagenet', include_top=False)
# for layer in model_resnet50_conv.layers: # freeze resnet50 for feature extraction
#     layer.trainable = False

#Create your own input format
input = Input(shape=(224,224,3),name = 'image_input')
#Use the generated model
output_resnet50_conv = model_resnet50_conv(input)

#Add the fully-connected layers
x = Flatten(name='flatten')(output_resnet50_conv)
x = Dense(101, activation='softmax', name='predictions')(x)

#Create your own model
model_resnet50 = Model(input=input, output=x)
model_resnet50.load_weights("model/resnet50_5_raw_rmspop_full.h5")

# train and save model
#model_resnet50.fit(x_train,y_train,batch_size=10,epochs=1,shuffle=True,verbose=2) #validation_data=(x_test, y_test)
model_resnet50.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.000001), #default lr 0.001
                       metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
model_resnet50.fit_generator(generate_xy('own-data/food101_n80800_train_r224x224x3_raw.h5',batch_size=50),
                             validation_data=generate_xy('own-data/food101_n20200_test_r224x224x3_raw.h5',batch_size=101), validation_steps=200,
                             steps_per_epoch=1616, epochs=1, verbose=1, shuffle=True) # steps_per_epoch=data/batch size

model_resnet50.save('model/resnet50_1.h5')
#with open("model/resnet50.json", "w") as json_file:
#    json_file.write(model_resnet50.to_json())
#model_resnet50.save_weights("model/resnet50.h5")

# # test
# stat = model_resnet50.evaluate_generator(generate_xy('own-data/food101_n20200_test_r224x224x3_norm.h5',batch_size=101),steps=200)
# print("loss:", stat[0])
# print("top 1 test accuracy:", stat[1])
# print("top 5 test accuracy:", stat[2])
