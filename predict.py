from __future__ import absolute_import, division, print_function

import os, sys

# machine learning
import tensorflow.keras as keras

# helper libraries
import numpy as np

# internal imports
from model import get_model
from data import wav2mfcc

if __name__ == "__main__":
  fn = "dataset/cats_dogs/dog_barking_0.wav"
  if len(sys.argv) > 1:
    fn = sys.argv[1]

  model = get_model()
  model.load_weights("checkpoint/10.hdf5")
  
  Xnew = wav2mfcc(fn).reshape((1, 20, 11, 1))
  ynew = model.predict_classes(Xnew)
  classes = ["cat", "dog"]
  print("predication:", classes[ynew[0]])