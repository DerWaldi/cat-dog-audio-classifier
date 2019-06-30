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

Xnew = wav2mfcc("dataset/cats_dogs/dog_barking_0.wav").reshape((1, 20, 11, 1))
ynew = model.predict_classes(Xnew)
print("dog predication:", ynew)