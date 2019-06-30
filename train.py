from __future__ import absolute_import, division, print_function

import os

# machine learning
import tensorflow.keras as keras

# helper libraries
import numpy as np

# internal imports
from config import BATCH_SIZE
from utils import TrainValTensorBoard
from model import get_model
from data import generate_datasets, wav2mfcc

train_dataset, steps_per_epoch, valid_dataset = generate_datasets()

model = get_model()  

tensorboardCallback = TrainValTensorBoard(write_graph=False)

if not os.path.exists("checkpoint/"):
  os.makedirs("checkpoint/")
    
saverCallback = keras.callbacks.ModelCheckpoint(
    "checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5", 
    monitor='val_loss', 
    verbose=0, save_best_only=False, 
    save_weights_only=False, 
    mode='auto', period=1)

model.fit(train_dataset, 
    validation_data=valid_dataset, 
    epochs=10, 
    steps_per_epoch=steps_per_epoch, 
    verbose=1, 
    callbacks=[tensorboardCallback, saverCallback])

# predict
Xnew = wav2mfcc("dataset/cats_dogs/cat_1.wav").reshape((1, 20, 11, 1))
ynew = model.predict_classes(Xnew)
print("cat predication:", ynew)
