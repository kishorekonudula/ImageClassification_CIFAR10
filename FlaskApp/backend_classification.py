import flask
from flask import request, render_template
from flask import Flask
import numpy as np
import pandas as pd
from copy import deepcopy
from keras.models import load_model
from PIL import Image
import pickle

# Initialize the app
#app = flask.Flask(__name__)
app = flask.Flask(__name__)

def pickleSomething(data, path, pickle_filename):
    with open(path+pickle_filename, 'wb') as p:
        pickled = pickle.dump(data, p)
    return pickled

def unpickleSomething(path, pickle_filename):
    with open(path+pickle_filename, 'rb') as p: 
        unpickled = pickle.load(p)
        return unpickled

def loadConvNetModel():
	model_path = "~/ConvNet/ModelInfo/"
	model = load_model(model_path+'model_1.h5')
	return model 

def getClassName(c):
	class_names = ['Plane', 'Car', 'Bird', 'Cat', 
	'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
	return class_names[c]

def testImageArray():
	path = "~/FlaskStuff/ImageArray/"
	pickle_filename = "x_test.p"
	test_array = unpickleSomething(path, pickle_filename)
	return test_array


@app.route("/")
def viz_page():
    with open("frontend_classification.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/img1")
def getImageClass1():
	# Plane
	index = 3
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name


@app.route("/img2")
def getImageClass2():
	# Car
	index = 9
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name
	#return flask.jsonify(img_class_name)

@app.route("/img3")
def getImageClass3():
	# Bird
	index = 67
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img4")
def getImageClass4():
	# Cat
	index = 0
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name


@app.route("/img5")
def getImageClass5():
	# Deer
	index = 22
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img6")
def getImageClass6():
	# Dog
	index = 22
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img7")
def getImageClass7():
	# Frog
	index = 5
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img8")
def getImageClass8():
	# Horse
	index = 17
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img9")
def getImageClass9():
	# Ship
	index = 2
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img10")
def getImageClass10():
	# Truck
	index = 11
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name


@app.route("/img11")
def getImageClass11():
	# Plane
	index = 3
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name


@app.route("/img22")
def getImageClass22():
	# Car
	index = 9
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name
	#return flask.jsonify(img_class_name)

@app.route("/img33")
def getImageClass33():
	# Bird
	index = 67
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img44")
def getImageClass44():
	# Cat
	index = 0
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name


@app.route("/img55")
def getImageClass55():
	# Deer
	index = 22
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img66")
def getImageClass66():
	# Dog
	index = 87
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img77")
def getImageClass77():
	# Frog
	index = 5
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img88")
def getImageClass88():
	# Horse
	index = 17
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img99")
def getImageClass99():
	# Ship
	index = 2
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

@app.route("/img1010")
def getImageClass1010():
	# Truck
	index = 11
	arr = testImageArray()
	img = arr[index:index+1, :, :, :]  
	# Apply model
	model = loadConvNetModel()
	img_class = model.predict_classes(img)
	img_class_name = getClassName(img_class)
	return img_class_name

#============================================


#--------- RUN WEB APP SERVER ------------#

if __name__ == '__main__' :
	app.run()


