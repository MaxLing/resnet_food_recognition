
# coding: utf-8

# In[1]:


# %load model_predict.py
import h5py
import numpy as np
import cv2
from keras import models
import matplotlib.pyplot as plt


with open('ETHZ-FOOD-101/food-101/meta/classes.txt') as f:
    classes = f.read().splitlines()

with h5py.File('own-data/food101_n20200_test_r224x224x3_raw.h5', 'r') as hf:
    x = np.array(hf["image"][:100], dtype=np.uint8).reshape((-1, 224, 224, 3))
    y = np.eye(101, dtype=np.uint8)[hf["category"][:100].reshape((-1,))]


model = models.load_model('model/resnet50_9_raw_rmspop.h5')
y_pred = model.predict(x, batch_size=20)

idx_true = np.argmax(y,axis=1)
idx_pred = np.argmax(y_pred,axis=1)

for i in range(100):
    # for j in range(5):
    #     print("the image is: ", classes[a[i][j]])
    a = np.argsort(y_pred[i])
    a = a[::-1] #BGR to RGB
    # for j in range(5):
    #     print("the image is: ", classes[a[j]],y_pred[i][a[j]])
    plt.figure(figsize=(14,4))
    ax = plt.gca()
    plt.imshow(x[i,:,:,:].reshape((224,224,3))[:,:,::-1],extent=[0,.5,0,.5])
    table_vals = []
    for j in range(5):
        table_vals.append([classes[a[j]], y_pred[i][a[j]]])
    col_labels = ['category', 'probability']
    row_labels = ['1', '2', '3', '4', '5']
    table = plt.table(cellText=table_vals, colWidths=[0.4 for x in col_labels],
                         rowLabels=row_labels,colLabels=col_labels,
                         loc="right", cellLoc='center',bbox=[1.2, 0.1, 0.8, 0.8])
    cell_dict = table.get_celld()
    table.set_fontsize(22)
    plt.title(classes[idx_true[i]])
    plt.show()




