# -*- coding: utf-8 -*-
"""
Created on Sat Sept 15 17:43:10 2019

@author: AI Voyagers
"""
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import os

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
modelname   = 'Veg_Classifier_Resnet'
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

optmz       = Adam(lr=0.001)


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

# define the deep learning model
def resLyr(inputs,numFilters=16,kernelSz=3,strides=1,activation='relu',batchNorm=True,convFirst=True,lyrName=None):
    
    convLyr = Conv2D(numFilters, kernel_size=kernelSz, strides=strides, padding='same',kernel_initializer='he_normal',
    kernel_regularizer=l2(1e-4), name=lyrName+'_conv' if lyrName else None)
    x = inputs
    if convFirst:
        x = convLyr(x)
        if batchNorm:
            x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
    else:
        if batchNorm:
            x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
        x = convLyr(x)
    return x


def resBlkV1(inputs,
             numFilters=16,
             numBlocks=3,
             downsampleOnFirst=True,
             names=None):
    
    x = inputs
    for run in range(0,numBlocks):
        strides = 1
        blkStr = str(run+1)
        if downsampleOnFirst and run == 0:
            strides = 2
            
        y = resLyr(inputs=x, numFilters=numFilters, strides=strides,lyrName=names+'_Blk'+blkStr+'_Res1' if names else None)
        
        y = resLyr(inputs=y,   numFilters=numFilters, activation=None, lyrName=names+'_Blk'+blkStr+'_Res2' if names else None)
        
        if downsampleOnFirst and run == 0:
            x = resLyr(inputs=x, numFilters=numFilters,kernelSz=1,
                       strides=strides,activation=None,batchNorm=False,
                       lyrName=names+'_Blk'+blkStr+'_lin' if names else None)
        
        x = add([x,y], name=names+'_Blk'+blkStr+'_add' if names else None)
        
        x = Activation('relu', name=names+'_Blk'+blkStr+'_relu' if names else None)(x)
    return x
    

def createResNetV1(inputShape=(128,128,3),
                   numClasses=5):
    
    inputs = Input(shape=inputShape)
    v = resLyr(inputs,    lyrName='Inpt')
    
    
    
    v = resBlkV1(inputs=v, numFilters=16, numBlocks=3, downsampleOnFirst=False, names='Stg1')
    
    v = resBlkV1(inputs=v, numFilters=32, numBlocks=3, downsampleOnFirst=True, names='Stg2')
    
    v = resBlkV1(inputs=v, numFilters=64, numBlocks=3,  downsampleOnFirst=True, names='Stg3')
    
    v = resBlkV1(inputs=v, numFilters=128, numBlocks=3,  downsampleOnFirst=True, names='Stg4')
    
    v = AveragePooling2D(pool_size=4, name='AvgPool')(v)
    
    v = Flatten()(v)
    
    outputs = Dense(numClasses, activation='softmax', kernel_initializer='he_normal')(v)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer=optmz,metrics=['accuracy'])
    return model

# Setup the models
trainmodel       = createResNetV1()  # This is meant for training
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

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

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
#callbacks_list  = [checkpoint,csv_logger,LRScheduler,early_stopping]
callbacks_list  = [checkpoint,csv_logger,LRScheduler]

 #.............................................................................
# Fit the model
# This is where the training starts
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=20,
                             horizontal_flip=True,
                             vertical_flip=True)
# ......................................................................
trainmodel.fit_generator(datagen.flow(training_dataset, training_lbl, batch_size=64),
                    validation_data=(validation_dataset, validation_lbl),
                    epochs=200, 
                    verbose=1,
                    steps_per_epoch=len(training_dataset)/64,
                   callbacks=callbacks_list)
                        
# ...................................................................

    
import pandas as pd

records     = pd.read_csv(csv_filepath)
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0.00,0.10,0.20,0.30,0.40])
plt.title('Loss value',fontsize=12)
ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.93,0.95,0.97,0.99])
plt.title('Accuracy',fontsize=12)
plt.show()


from tensorflow.keras.utils import plot_model

plot_model(trainmodel, 
           to_file=pdf_filepath, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')

