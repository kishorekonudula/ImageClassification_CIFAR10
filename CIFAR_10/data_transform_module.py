import pickle
import numpy as np


def pickleSomething(data, path, pickle_filename):
    with open(path+pickle_filename, 'wb') as p:
        pickled = pickle.dump(data, p)
    return pickled

def unpickleSomething(path, pickle_filename):
    with open(path+pickle_filename, 'rb') as p: 
        unpickled = pickle.load(p)
        return unpickled

def array2pixFeatures(arr):

    pixel_features_all_img = []
	
    for i in range(0, arr.shape[0]):
		
        img_array = arr[i]
    
        r_channel = img_array[0, :, :]
        g_channel = img_array[1, :, :]
        b_channel = img_array[2, :, :]

        r_vector = np.transpose(r_channel).reshape(1, 32*32)
        g_vector = np.transpose(g_channel).reshape(1, 32*32)
        b_vector = np.transpose(b_channel).reshape(1, 32*32)

        rgb_vector = list(r_vector[0]) + list(g_vector[0]) + list(b_vector[0])
        

        pixel_features_all_img.append(rgb_vector)

    return np.array(pixel_features_all_img)

