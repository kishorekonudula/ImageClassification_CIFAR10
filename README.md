### Classification of the CIFAR-10 dataset

**About the data:**  
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6000 images per class.  
There are 50,000 training images and 10,000 test images.


**Base Directory:** `ImageClassification/`  

**Sub-directories:**

* `CIFAR_10/`  
Download, uncompress, extract: `cifar10_download.py`  
Display RGBs as PNG: `display_cifar10.ipynb`  
Un/pickel data, reshape image arrays: `data_transform_module.py`  

* `CIFAR_10/Features/`  
Compute features: `pix_features.py`, `pca_features.py`

* `CIFAR_10/DisplayImages/`  
`display_images.ipynb`: accesses sql database on aws, extracts image information and displays as RGBs.

* `ConvNet/`  
 Codes for convolutional neural modeling (note only the final model is included)

* `KNN/`  
Codes for k-nearest-neighbor modeling

* `SVM/`  
Codes for SVM modeling

* `FlaskApp/`  
Image classification demo

* `Slides/`  
Presentation slides
