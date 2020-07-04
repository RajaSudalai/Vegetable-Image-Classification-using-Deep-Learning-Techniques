# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:43:10 2019

@author: AI Voyagers
"""
import numpy as np
import sklearn.metrics as metrics
import h5py
import tensorflow
import re

model_dir = "models/"
model_list = os.listdir(model_dir)
print ("models : len - ", len(model_list), model_list)
# .......................................................................
# Loading the datasets from the stored dataset
with h5py.File('dataset_models/TestingDataset.hdf5', 'r') as hf:
    testing_dataset = hf['TestingDataset'][:]


with h5py.File('dataset_models/TestingLbl.hdf5', 'r') as hf:
    testing_lbl = hf['TestingLbl'][:]
#---------------------------------------------------------------------------

# load models from file
def predict_all_models(model_dir, model_list):
    all_models = list()
    scores_dict = {}
    for i, model_name in enumerate(model_list, start=1) :
        # load model from file
        #model = joblib.load(model_dir + model_name)
        model =  tensorflow.keras.models.load_model(model_dir + model_name)
        # add to list of members
        all_models.append(model)
        print(f"[{i:02d}] model loaded : {model_name}")
        
        
        # Make classification on the test dataset
        predicts    = model.predict(testing_dataset)
        
        # Prepare the classification output
        # for the classification report
        predout     = np.argmax(predicts,axis=1)
        testout     = np.argmax(testing_lbl,axis=1)
        labelname   = ['Brocolli','Cucumber','Pepper', 'Pumpkin','Tomato']
        # the labels for the classfication report
        
        testScores  = metrics.accuracy_score(testout,predout)
        confusion   = metrics.confusion_matrix(testout,predout)
        
        
        print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
        scores_dict[model_name] = testScores*100
        print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
        print(confusion)
    return scores_dict
 
# load all models
scores_dict = predict_all_models(model_dir, model_list)
print('Predcited %d models' % len(scores_dict))
print(scores_dict)

print(' ----------------------------------------------- ' )
print(' Accuracy Report for all the generated models ' )
print(' ----------------------------------------------- ' )
for i, model_name in enumerate(scores_dict, start=1) :
    #to remove the extension
    model_name_without_extension = re.split(r'[.]', model_name)
    print( model_name_without_extension[0] , '====>', scores_dict[model_name])
    


   
#