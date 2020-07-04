# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:43:10 2019

@author: AI Voyagers
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import os
from tensorflow.keras.utils import plot_model


#makedirs('models')
if not os.path.exists('models'):
    os.makedirs('models')
    
#makedirs('pdfmodels')
if not os.path.exists('pdfmodels'):
    os.makedirs('pdfmodels')
    
#makedirs('csvlogger')
if not os.path.exists('csvlogger'):
    os.makedirs('csvlogger')
    
#define file name
modelname   = 'Veg_Classifier_CNN' 
model_filepath    = 'models/'+modelname + ".hdf5"
csv_filepath = 'csvlogger/'+modelname +'.csv'
pdf_filepath = 'pdfmodels/'+ modelname +'.pdf'


# Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'



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

# .............................................................................

# fix random seed for reproducibility
seed        = 29
np.random.seed(seed)

opt = 'RMSprop'


# define the deep learning model
def createModel():
   
    inputs= Input(shape=(128,128,3))
    x = Conv2D(48, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3),padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, (3, 3),padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3),padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=inputs,outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model
                           # Setup the models
trainmodel       = createModel() # This is meant for training
trainmodel.summary()

# .............................................................................


def lrSchedule(epoch):
    lr  = 1e-3
    
    if epoch > 160:
        lr  *= 0.5e-3
        
    elif epoch > 140:
        lr  *= 1e-3
        
    elif epoch > 120:
        lr  *= 1e-2
        
    elif epoch > 80:
        lr  *= 1e-1
        
    print('Learning rate: ', lr)
    
    return lr

LRScheduler     = LearningRateScheduler(lrSchedule)

# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest validation accuracy

checkpoint      = ModelCheckpoint(model_filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

# Log the epoch detail into csv
csv_logger      = CSVLogger(csv_filepath)
callbacks_list  = [checkpoint,csv_logger,LRScheduler]


# .............................................................................

# Fit the model
# This is where the training starts

datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=20,
                             horizontal_flip=True,
                             vertical_flip=False)
# ......................................................................
trainmodel.fit_generator(datagen.flow(training_dataset, training_lbl, batch_size=64),
                    validation_data=(validation_dataset, validation_lbl),
                    epochs=200, 
                    verbose=1,
                    steps_per_epoch=len(training_dataset)/64,
                    callbacks=callbacks_list)

# ..................................................................
#Plot the val accuracy and loss pdf results
import pandas as pd

records     = pd.read_csv(csv_filepath)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*2)
#plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0,0.20,0.40,0.60,0.80,1.00,1.20,1.40,1.60,1.80,2.0])
plt.title('Loss value',fontsize=12)
ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9,1.0])
plt.title('Accuracy',fontsize=12)
plt.show()



#Plot and Export the pdf model ('models')
plot_model(trainmodel, 
           to_file=pdf_filepath, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')


