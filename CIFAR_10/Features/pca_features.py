"""
Compute pricipal components of image pixels
using PCA
"""


from sklearn.decomposition import PCA
from os import path
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle

sys.path.append(path.abspath('~/CIFAR_10/'))
import data_transform_module as dtm

# path where pca features will be written out
feature_path = "~/CIFAR_10/Features/ImageFeatures/"
image_data_path = "~/CIFAR_10/Features/ImageData/"


def createPixFeatures():
	# training data
	img_trn_arr = dtm.unpickleSomething(image_data_path, 'img_train.p')
	Xtrn = dtm.array2pixFeatures(img_trn_arr)
	# test data
	img_test_arr = dtm.unpickleSomething(image_data_path, 'img_test.p')
	Xtest = dtm.array2pixFeatures(img_test_arr)


def applyPCA(X, path, prefix_filename):
    
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    
    energy = []
    for k in prin_comp:
        print(k)
        pickle_filename = prefix_filename+str(k)+".p"
        pca = PCA(n_components=k, whiten=False).fit(X_minmax)
        eigen_energy = sum(pca.explained_variance_ratio_)
        energy.append(eigen_energy)
        X_pca = pca.transform(X_minmax)
        
        dtm.pickleSomething(X_pca, path, pickle_filename)
    
    return energy

prin_comp = [10, 25, 50, 75, 100, 150, 200, 250, 300]
#apply pca on training data
trn_energy  = applyPCA(Xtrn, path, "trn_pca", prin_comp)
#apply pca on test data
test_energy  = applyPCA(Xtest, path, "test_pca", prin_comp)


