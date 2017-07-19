"""
This code converts each CIFAR-10 image into a 1D vector.

The downloaded CIFAR images are in nested multidimensional
format. ~/CIFAR_10/ImageData/img*_.p 
E.g. test images: img_test.p --> 10000x32x32x3

The R, G, B channels of each image are vecotorized and concatenated.
The vecorized images are stored in a numpy array.
E.g. Traning data
	Xtrn.shape (50000, 1, 3072)
	Each row in Xtrn (1, 3072), a vectorized image.
"""

# python modules
from os import path
import sys
import numpy as np
import pickle

# my modules
sys.path.append(path.abspath('~/CIFAR_10/'))
import data_transform_module as dtm

# paths
# path where pca features will be written out
image_data_path = "~/CIFAR_10/ImageData/"
feature_path = "~/CIFAR_10/ImageFeatures/"


# training data
img_trn_arr = dtm.unpickleSomething(image_data_path, 'img_train.p')
Xtrn = dtm.array2pixFeatures(img_trn_arr)
pickle_filename = "trn_rgbpix.p"
dtm.pickleSomething(Xtrn, feature_path, pickle_filename)

# test data
img_test_arr = dtm.unpickleSomething(image_data_path, 'img_test.p')
Xtest = dtm.array2pixFeatures(img_test_arr)
pickle_filename = "test_rgbpix.p"
dtm.pickleSomething(Xtest, feature_path, pickle_filename)




