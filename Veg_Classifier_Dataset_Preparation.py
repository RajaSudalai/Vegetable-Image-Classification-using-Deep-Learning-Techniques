# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:05:14 2019

@author: AIVoyagers
"""

from imutils import paths
import cv2
import os
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


#makedirs('models')
if not os.path.exists('dataset_models'):
    os.makedirs('dataset_models')
    
    
imagePaths = list(paths.list_images("datasets"))
data = []
labels = []
print("Loading Images")
# loop over the image paths
# extract the class label from the filename, load the image, and
# resize it to be a fixed 128x128 pixels, ignoring aspect ratio
# update the data and labels lists, respectively
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128,128))
    data.append(image)
    labels.append(label)

print('Statistics of Images')
print("Size of Images"+ str(len(data)))  
print("Size of labels"+ str(len(labels)))


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder() 
labels = le.fit_transform(labels)
labels = to_categorical(labels,5)

# partition the data into training and testing splits using 85%(70% for training, 15% for testing) of
# the data for training and the remaining 155% for testing

print('Training/testing  dataset split')
(trainingdataset, testingdataset, traininglbl, testinglbl) = train_test_split(data, labels, stratify = labels, test_size=0.15, random_state=42)
print('Size of training dataset-'+ str(len(trainingdataset)))
print("Size of training label- "+ str(len(traininglbl)))
print("Size of training dataset- "+ str(len(testingdataset)))
print("Size of training label- "+ str(len(testinglbl)))

print('Shape of training dataset-')
print(trainingdataset.shape)
print("Shape of training label- ")
print(traininglbl.shape)
print("Shape of testingdataset- ")
print(testingdataset.shape)
print("Shape of testinglbl - ")
print(testinglbl.shape)


print("Training/validation dataset split")
(trainingdataset, validationdataset, traininglbl, validationlbl) = train_test_split(trainingdataset,  traininglbl, stratify = traininglbl, test_size=0.15, random_state=42)
print("Size of training dataset- "+ str(len(trainingdataset)))
print("Size of training label -"+ str(len(traininglbl)))
print("Size of validation  dataset -"+ str(len(validationdataset)))
print("Size of validation  label -"+ str(len(validationlbl)))


print('Shape of training dataset-')
print(trainingdataset.shape)
print("Shape of training label- ")
print(traininglbl.shape)
print("Shape of validationdataset dataset- ")
print(validationdataset.shape)
print("Shape of validationdataset label- ")
print(validationlbl.shape)

#Storing the data set in h5 for model generation purposes.

with h5py.File('dataset_models/TrainingDataset.hdf5', 'w') as hf:
    hf.create_dataset("TrainingDataset",  data=trainingdataset)
    
with h5py.File('dataset_models/TrainingLbl.hdf5', 'w') as hf:
    hf.create_dataset("TrainingLbl",  data=traininglbl)
    
with h5py.File('dataset_models/ValidationDataset.hdf5', 'w') as hf:
    hf.create_dataset("ValidationDataset",  data=validationdataset)
    
with h5py.File('dataset_models/ValidationLbl.hdf5', 'w') as hf:
    hf.create_dataset("ValidationLbl",  data=validationlbl)
    
with h5py.File('dataset_models/TestingDataset.hdf5', 'w') as hf:
    hf.create_dataset("TestingDataset",  data=testingdataset)
    
with h5py.File('dataset_models/TestingLbl.hdf5', 'w') as hf:
    hf.create_dataset("TestingLbl",  data=testinglbl)



