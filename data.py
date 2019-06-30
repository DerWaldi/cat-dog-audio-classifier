from __future__ import absolute_import, division, print_function

import os
from time import time

# machine learning
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.metrics

# keras
from tensorflow.keras.callbacks import TensorBoard

# helper libraries
import numpy as np
import pandas as pd

# domain specific libraries
import librosa

# internal imports
from config import BATCH_SIZE

# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):    
  wave, sr = librosa.load(file_path, mono=True, sr=None)
  wave = wave[::3]
  mfcc = librosa.feature.mfcc(wave, sr=16000)

  # If maximum length exceeds mfcc lengths then pad the remaining ones
  if (max_len > mfcc.shape[1]):
    pad_width = max_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

  # Else cutoff the remaining parts
  else:
    mfcc = mfcc[:, :max_len]
  
  return mfcc.reshape((20, max_len, 1)).astype(np.float32)

#wav2mfcc("C:/Workspace/datasets/cats_dogs/cats_dogs/cat_1.wav").shape

def _parse_data(input):
  audio_features = tf.compat.v1.py_func(wav2mfcc, [input], tf.float32)
  audio_features.set_shape([20, 11, 1])
  return audio_features

def generate_datasets():  
  # https://www.kaggle.com/mmoreaux/audio-cats-and-dogs/downloads/audio-cats-and-dogs.zip
  fn_train_test_csv = "dataset/train_test_split.csv"
  df = pd.read_csv(fn_train_test_csv)

  # Load Data into list
  path_audio = "dataset/cats_dogs/"

  # get col as array from dataframe
  fn_test_cats = df["test_cat"]
  # filter invalid entries and make full filename
  fn_test_cats = list(map(lambda c: path_audio + c, filter(lambda c: isinstance(c, str), fn_test_cats)))

  fn_test_dogs = df["test_dog"]
  fn_test_dogs = list(map(lambda c: path_audio + c, filter(lambda c: isinstance(c, str), fn_test_dogs)))

  fn_train_cats = df["train_cat"]
  fn_train_cats = list(map(lambda c: path_audio + c, filter(lambda c: isinstance(c, str), fn_train_cats)))

  fn_train_dogs = df["train_dog"]
  fn_train_dogs = list(map(lambda c: path_audio + c, filter(lambda c: isinstance(c, str), fn_train_dogs)))

  x_train = fn_train_cats + fn_train_dogs
  y_train = list(map(lambda c: 0, fn_train_cats)) + list(map(lambda c: 1, fn_train_dogs))

  x_test = fn_test_cats + fn_test_dogs
  y_test = list(map(lambda c: 0, fn_test_cats)) + list(map(lambda c: 1, fn_test_dogs))

  # create the training datasets
  dx_train = tf.data.Dataset.from_tensor_slices(x_train).map(_parse_data)
  # apply a one-hot transformation to each label for use in the neural network
  dy_train = tf.data.Dataset.from_tensor_slices(y_train).map(lambda z: tf.one_hot(z, 2))
  # zip the x and y training data together and shuffle, batch etc.
  train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)

  # do the same operations for the validation set
  dx_valid = tf.data.Dataset.from_tensor_slices(x_test).map(_parse_data)
  dy_valid = tf.data.Dataset.from_tensor_slices(y_test).map(lambda z: tf.one_hot(z, 2))
  valid_dataset = tf.data.Dataset.zip((dx_valid, dy_valid)).batch(30)

  steps_per_epoch = len(x_train) // BATCH_SIZE

  return train_dataset, steps_per_epoch, valid_dataset