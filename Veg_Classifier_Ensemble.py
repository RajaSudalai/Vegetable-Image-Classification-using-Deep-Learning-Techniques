# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:16:27 2019

@author: AIVoyagers
"""


import os
import tensorflow
import h5py
import sklearn.metrics as metrics
import numpy as np
from numpy import array
import numpy
from numpy import argmax
from matplotlib import pyplot




 
model_dir = "models/"
model_list = os.listdir(model_dir)
print ("models : len - ", len(model_list), model_list)

# ---------------------------------------------------------------------------
# Loading the datasets from the stored dataset
with h5py.File('dataset_models/TestingDataset.hdf5', 'r') as hf:
    testing_dataset = hf['TestingDataset'][:]


with h5py.File('dataset_models/TestingLbl.hdf5', 'r') as hf:
    testing_lbl = hf['TestingLbl'][:]
    
with h5py.File('dataset_models/TrainingDataset.hdf5', 'r') as hf:
    training_dataset = hf['TrainingDataset'][:]
    print("Size of training dataset- "+ str(len(training_dataset)))
    
with h5py.File('dataset_models/TrainingLbl.hdf5', 'r') as hf:
    training_lbl = hf['TrainingLbl'][:]
    print("Size of training dataset- "+ str(len(training_lbl)))
#---------------------------------------------------------------------------

# load models from file
def load_all_models(model_dir, model_list):
    all_models = list()
    for i, model_name in enumerate(model_list, start=1) :
        # load model from file
        #model = joblib.load(model_dir + model_name)
        model =  tensorflow.keras.models.load_model(model_dir + model_name)
        # add to list of members
        all_models.append(model)
        print(f"[{i:02d}] model loaded : {model_name}")
    return all_models
 
# load all models
members = load_all_models(model_dir, model_list)
print('Loaded %d models' % len(members))


#---------------------------------------------------------------------------
# evaluate standalone models on test dataset before ensemble
scores = list()
for i, model in enumerate(members):
    
    predicts = model.predict(testing_dataset)
    predout     = np.argmax(predicts,axis=1)
    testout     = np.argmax(testing_lbl,axis=1)
    testScores  = metrics.accuracy_score(testout,predout)
    confusion   = metrics.confusion_matrix(testout,predout)
    print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
    scores.append(testScores)
    
print("Standard deviation scores for all model: %.2f%%" % (np.std(scores)))

    # plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, scores)
pyplot.show()

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)

    return result

# evaluate ensemble model
def evaluate_members(members, testX, testy):
    # make prediction
    yhat = ensemble_predictions(members, testX)
    testout     = np.argmax(testy,axis=1)
    testScores  = metrics.accuracy_score(testout,yhat)
    # calculate accuracy
    return testScores


# evaluate ensemble
ensemble_score = evaluate_members(members, testing_dataset, testing_lbl)
print("Best accuracy of average ensemble (on testing dataset): %.2f%%" % (ensemble_score*100))
