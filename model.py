from __future__ import absolute_import, division, print_function

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

import tensorflow.keras.backend as K

feature_dim = [20, 11]
num_classes = 2

def recall(y_true, y_pred):
  """Recall metric.

  Only computes a batch-wise average of recall.

  Computes the recall, a metric for multi-label classification of
  how many relevant items are selected.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision(y_true, y_pred):
  """Precision metric.

  Only computes a batch-wise average of precision.

  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1(y_true, y_pred):
  """F1 Score metric."""
  prec = precision(y_true, y_pred)
  rec = recall(y_true, y_pred)
  return 2*((prec*rec)/(prec+rec+K.epsilon()))

def get_model():
  model = Sequential()
  model.add(Conv2D(64, (3, 3), input_shape=(feature_dim[0], feature_dim[1], 1), padding='same',))
  model.add(Conv2D(32, (3, 3), input_shape=(feature_dim[0], feature_dim[1], 1), padding='same',))
  model.add(Flatten(data_format=None))
  model.add(Dense(1024, activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(2, activation='softmax'))
  model.compile(
      loss=keras.losses.categorical_crossentropy, 
      optimizer=keras.optimizers.Adam(),
      metrics=['accuracy', recall, precision, f1]
  )
  return model