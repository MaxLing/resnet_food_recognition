# food-101-resnet

This is a Project for CIS519 Introduction to Machine Learning at Upenn. It implements ResNet50 in keras and applies transfer learning from Imagenet to recognize food. The best model achieves 77.25% Top1 and 92.90% Top5 testing accuracy after just 9 training epochs which takes only 5 hour.

Data Preprocess:
Download ETHZ-FOOD-101 (https://www.vision.ee.ethz.ch/datasets_extra/food-101/) or UPMC-FOOD-101, modify path and run data_preprocess.py to generate cropped 224x224x3 image data. The image data are stored and splited in h5 file(Each categories has 200 testing and 800 training images). You can optionally normalize image by grey world method and histogram equalization. 

Last Layer Training:


Full Model Training:

Generated Input for categories:
