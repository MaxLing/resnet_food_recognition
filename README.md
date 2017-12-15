# food-101-resnet

This is a Project for CIS519 Introduction to Machine Learning at Upenn. It implements ResNet50 in keras and applies transfer learning from Imagenet to recognize food. The best model achieves 77.25% Top1 and 92.90% Top5 testing accuracy after just 9 training epochs which takes only 5 hour.

Data Preprocess:
Download ETHZ-FOOD-101 (https://www.vision.ee.ethz.ch/datasets_extra/food-101/) or UPMC-FOOD-101, modify path and run data_preprocess.py to generate cropped 224x224x3 image data. The image data are stored and splited in h5 file(Each categories has 200 testing and 800 training images). You can optionally normalize image by grey world method and histogram equalization. 

Last Layer Training:
ResNet50 model should be initialized with Imagenet weights, replace last layer with 101 output softmax dense layer and freeze previous layers. After first saving, the model initialization should be commented and the model loading should be uncommented. A example is last_training.ipynb. 5 epochs with init learning rate 0.001 is good enough.

Full Model Training:
Later un-freeze previous layers and fine-tune the model. A example is full_trainning.ipynb. 4 epochs with init learning rate 0.00001 is good enough. Please note batch size is limited by GPU memory, GTX1050 is used for last layer training and handles 101 batch size, while Tesla K80 is used for full model training and handles 50 batch size.

Visualization:
Using the saved model, we can run generated_input.py to generate categorical input images or run model_predict.py to show top5 predictions of testing images.
