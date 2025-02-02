# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:44:18 2019

@author: AI Voyagers
"""


from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import os
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import h5py
from tensorflow.keras.utils import plot_model
    
#makedirs('models')
if not os.path.exists('models'):
    os.makedirs('models')
    
#makedirs('pdfmodels')
if not os.path.exists('pdfmodels'):
    os.makedirs('pdfmodels')
    

    
#define file name
modelname   = 'Veg_Classifier_CNN_Tuning_Add_Layer' 
model_filepath    = 'models/'+modelname + ".hdf5"
pdf_filepath = 'pdfmodels/'+ modelname +'.pdf'



def data():
    
    # ............................................................................
    # Loading the datasets from the stored dataset
    with h5py.File('dataset_models/TrainingDataset.hdf5', 'r') as hf:
        training_dataset = hf['TrainingDataset'][:]
    print("Size of training dataset- "+ str(len(training_dataset)))
    
    with h5py.File('dataset_models/TrainingLbl.hdf5', 'r') as hf:
        training_lbl = hf['TrainingLbl'][:]
    print("Size of training dataset- "+ str(len(training_lbl)))
    
    with h5py.File('dataset_models/ValidationDataset.hdf5', 'r') as hf:
        validation_dataset = hf['ValidationDataset'][:]
    print("Size of validation dataset- "+ str(len(validation_dataset)))
    
    with h5py.File('dataset_models/ValidationLbl.hdf5', 'r') as hf:
        validation_lbl = hf['ValidationLbl'][:]
    print("Size of validation dataset- "+ str(len(validation_lbl)))
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

      
    X_train = training_dataset
    y_train = training_lbl
    X_val = validation_dataset
    y_val = validation_lbl
    datagen.fit(X_train)
    return datagen, X_train, y_train, X_val, y_val



    

def model(datagen, X_train, y_train, X_val, y_val):
    
    inputs = Input(shape=(128,128,3))
    x = Convolution2D(48, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Convolution2D(64, (3, 3),padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Convolution2D(128, (3, 3),padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    model_choice = {{choice(['three', 'four'])}}
    if model_choice == 'four':
    
        x = Convolution2D({{choice([128, 256])}}, {{choice([3, 5])}}, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout({{uniform(0, 0.8)}})(x)

     
    x = Flatten()(x)
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(5, activation='softmax')(x)
    opt =optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    
          
    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=128),     epochs=100,
                        validation_data=(X_val, y_val))

    score, acc = model.evaluate(X_val, y_val, verbose=0)
  
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    
    datagen, X_train, y_train, X_val, y_val = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_val, y_val))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
  
    
    best_model.save(model_filepath)
    
    


    plot_model(best_model, 
           to_file=pdf_filepath, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')