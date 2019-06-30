from __future__ import absolute_import, division, print_function

import os
from time import time

# machine learning
import tensorflow as tf
import tensorflow.keras as keras

# keras
from tensorflow.keras.callbacks import TensorBoard

# https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
class TrainValTensorBoard(TensorBoard):
  def __init__(self, log_dir="logs/{}/".format(time()).replace(".", "-"), **kwargs):        
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)    
    # Make the original `TensorBoard` log to a subdirectory 'training'
    self.training_log_dir = os.path.join(log_dir, 'training')
    if not os.path.exists(os.path.join(self.training_log_dir, 'plugins/profile/')):
      os.makedirs(os.path.join(self.training_log_dir, 'plugins/profile/'))

    # Log the validation metrics to a separate subdirectory
    self.val_log_dir = os.path.join(log_dir, 'validation')
    if not os.path.exists(self.val_log_dir):
      os.makedirs(self.val_log_dir)
        
    super(TrainValTensorBoard, self).__init__(self.training_log_dir, **kwargs)

  def set_model(self, model):
    # Setup writer for validation metrics
    self.val_writer = tf.summary.FileWriter(self.val_log_dir)
    self.train_writer = tf.summary.FileWriter(self.training_log_dir)
    super(TrainValTensorBoard, self).set_model(model)

  def on_epoch_end(self, epoch, logs=None):
    # Pop the validation logs and handle them separately with
    # `self.val_writer`. Also rename the keys so that they can
    # be plotted on the same figure with the training metrics
    logs = logs or {}
    val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
    for name, value in val_logs.items():
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = "performance/epoch_" + name
      self.val_writer.add_summary(summary, epoch)
    self.val_writer.flush()
    
    train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    for name, value in train_logs.items():
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = "performance/epoch_" + name
      self.train_writer.add_summary(summary, epoch)
    self.train_writer.flush()

    # Pass the remaining logs to `TensorBoard.on_epoch_end`
    logs = {}#{('performance/' + k): v for k, v in logs.items() if not k.startswith('val_')}
    super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

  def on_train_end(self, logs=None):
    super(TrainValTensorBoard, self).on_train_end(logs)
    self.val_writer.close()        
    self.train_writer.close()