"""
Download CIFAR-10 from http://www.cs.toronto.edu/~kriz/
Data file: cifar-10-python.tar.gz

CIFAR-10 dataset:
- 60000 32x32 RGB images 
- 10 classes
- 6000 images per class. 
- 50000 training images
- 10000 test images. 

*.gz data file had five training batches and one test batch (10000 images in each batch).

Images are extracted and stored as mutidimensional array.

OUTPUT arrays: 
Training images: 'img_train.p'   Array size: (50000, 32, 32, 3)
Training labels: 'label_train.p' Array size: (50000, )
Test images:     'img_test.p'    Array size: (10000, 32, 32, 3)
Test labels:     'label_test.p'  Array size: (50000, )
"""

import os
from subprocess import call
import pickle
import copy
import numpy as np
import numpy
import sklearn


cifar_python_directory = os.path.abspath("cifar-10-batches-py")


def pickleSomething(data, path, pickle_filename):
	with open(path+pickle_filename, 'wb') as p:
		pickled = pickle.dump(data, p)
	return pickled

def unpickleSomething(path, pickle_filename):
	with open(path+pickle_filename, 'rb') as p: 
		unpickled = pickle.load(p)
	return unpickled


def download():
	print("")
	print("Downloading compressed image files ...")
	if not os.path.exists("cifar-10-python.tar.gz"):
		call(
			"wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
			shell=True )
		print("Downloading done.\n")
		print("")
	else:
		print("Dataset already downloaded. Did not download again.\n")
		print("")


def uncompress():
	print("")
	print("Extracting from compressed...")
	cifar_python_directory = os.path.abspath("cifar-10-batches-py")
	if not os.path.exists(cifar_python_directory):
		call( "tar -zxvf cifar-10-python.tar.gz",shell=True)
		print("Data extracted to {}.".format(cifar_python_directory))
		print("")
	else:
		print("Dataset already extracted. Did not extract again.\n")
		print("")

def writeImageArray():
	print("Converting pickle files to array ...")
	
	def unpickle(file):
		fo = open(file, 'rb')
		u = pickle._Unpickler(fo)
		u.encoding = 'latin1'
		p = u.load()
		batch_dictionary = p
		fo.close()
		return batch_dictionary
    

	def shuffle_data(data, labels):
		data, _, labels, _ = sklearn.model_selection.train_test_split(
        	data, labels, test_size=0.0, random_state=42)
		return data, labels

	def load_data(train_batches):
		"""
		load one batch of files at a time
		this is for training images
		"""
		data = []
		labels = []
        
		for data_batch_i in train_batches:
			d = unpickle(os.path.join(cifar_python_directory, data_batch_i))
            
			data.append(d['data'])
			labels.append(np.array(d['labels']))
            
        	# Merge training batches on their first dimension
			data = np.concatenate(data)
			labels = np.concatenate(labels)
			length = len(labels)

			data, labels = shuffle_data(data, labels)
		return data.reshape(length, 3, 32, 32), labels

	X, y = load_data(["data_batch_{}".format(i) for i in range(1, 6)])

	Xt, yt = load_data(["test_batch"])
    
	print("Data has been converted to numpy array.\n")
    
	print("Pickling numpy array ...")
    	
	image_path = "/Users/susmitadatta/Metis/Proj03/CIFAR_10/ImageData/"
	# pickleSomething(X,  image_path, 'img_train.p',)
	# pickleSomething(y,  image_path, 'label_train.p')
	# pickleSomething(Xt, image_path, 'img_test.p')
	# pickleSomething(yt, image_path, 'label_test.p')

	print("The pickled files are in {}".format(image_path))



def main():
	download()
	uncompress()
	writeImageArray()


if __name__ == "__main__":
	main()
	