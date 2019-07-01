from __future__ import absolute_import, division, print_function

import os
import pickle

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.optimizers.bohb import BOHB
from hpbandster.core.result import json_result_logger
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker

# internal imports
from config import BATCH_SIZE
from utils import TrainValTensorBoard
from model import get_model
from data import generate_datasets, wav2mfcc

import logging
logging.basicConfig(level=logging.DEBUG)

class KerasWorker(Worker):
  def __init__(self, N_train=8192, N_valid=1024, **kwargs):
    super().__init__(**kwargs)

    self.batch_size = 30

    img_rows = 20
    img_cols = 11
    self.num_classes = 2

    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    #if K.image_data_format() == 'channels_first':
    #  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #  self.input_shape = (1, img_rows, img_cols)
    #else:
    #  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #  self.input_shape = (img_rows, img_cols, 1)


    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    # zero-one normalization
    #x_train /= 255
    #x_test /= 255


    # convert class vectors to binary class matrices
    #y_train = keras.utils.to_categorical(y_train, self.num_classes)
    #y_test = keras.utils.to_categorical(y_test, self.num_classes)


    #self.x_train, self.y_train = x_train[:N_train], y_train[:N_train]
    #self.x_validation, self.y_validation = x_train[-N_valid:], y_train[-N_valid:]
    #self.x_test, self.y_test   = x_test, y_test

    self.input_shape = (img_rows, img_cols, 1)
    
    self.train_dataset, self.steps_per_epoch, self.valid_dataset = generate_datasets()


  def compute(self, config, budget, working_directory, *args, **kwargs):
    """
    Simple example for a compute function using a feed forward network.
    It is trained on the MNIST dataset.
    The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
    """

    self.train_dataset, self.steps_per_epoch, self.valid_dataset = generate_datasets()

    model = Sequential()

    model.add(Conv2D(config['num_filters_1'], kernel_size=(3,3),
        activation='relu',
        input_shape=self.input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if config['num_conv_layers'] > 1:
      model.add(Conv2D(config['num_filters_2'], kernel_size=(3, 3),
          activation='relu',
          input_shape=self.input_shape))
      model.add(MaxPooling2D(pool_size=(2, 2)))	

    model.add(Dropout(config['dropout_rate']))
    model.add(Flatten())
    model.add(Dense(config['num_fc_units'], activation='relu'))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(self.num_classes, activation='softmax'))


    if config['optimizer'] == 'Adam':
      optimizer = keras.optimizers.Adam(lr=config['lr'])
    else:
      optimizer = keras.optimizers.SGD(lr=config['lr'], momentum=config['sgd_momentum'])

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy'])

    model.fit(self.train_dataset,
        steps_per_epoch=self.steps_per_epoch,
        epochs=int(budget),
        verbose=0,
        validation_data=self.valid_dataset)

    train_score = model.evaluate(self.train_dataset, steps=1, verbose=0)
    val_score = model.evaluate(self.valid_dataset, steps=1, verbose=0)
    #test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

    #import IPython; IPython.embed()
    return ({
      'loss': 1-val_score[1], # remember: HpBandSter always minimizes!
      'info': {
        #'test accuracy': float(test_score[1]),
        'train accuracy': float(train_score[1]),
        'validation accuracy': float(val_score[1]),
        'number of parameters': float(model.count_params()),
      }
    })


  @staticmethod
  def get_configspace():
    """
    It builds the configuration space with the needed hyperparameters.
    It is easily possible to implement different types of hyperparameters.
    Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
    :return: ConfigurationsSpace-Object
    """
    cs = CS.ConfigurationSpace()

    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

    # For demonstration purposes, we add different optimizers as categorical hyperparameters.
    # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
    # SGD has a different parameter 'momentum'.
    optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

    sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

    cs.add_hyperparameters([lr, optimizer, sgd_momentum])



    num_conv_layers =  CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=2, default_value=1)

    num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
    num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
    num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)

    cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])


    dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
    num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

    cs.add_hyperparameters([dropout_rate, num_fc_units])


    # The hyperparameter sgd_momentum will be used,if the configuration
    # contains 'SGD' as optimizer.
    cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    # You can also use inequality conditions:
    cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
    cs.add_condition(cond)

    return cs

if __name__ == "__main__":
  # config
  working_dir="bohb/"
  run_id = 'codel_01'  # Unique (at run time) identifier that has to be the same for the main process and all workers
  max_num_epochs = 3  # will be used within training of each model (=max budget)
  min_num_epochs = 1
  n_iterations = 3 # The number of iterations to run

  # enable live logging so a run can be canceled at any time and we can still recover the results
  result_logger = json_result_logger(directory=working_dir, overwrite=False)

  # start a nameserver for communication
  NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=0, working_directory=working_dir)
  ns_host, ns_port = NS.start()

  worker = KerasWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id, timeout=120)
  worker.run(background=True)

  bohb = BOHB(
    configspace = worker.get_configspace(),
    working_directory=working_dir,
    run_id = run_id,
    min_budget=min_num_epochs, 
    max_budget=max_num_epochs,
    host=ns_host,
    nameserver=ns_host,
    nameserver_port = ns_port,
    result_logger=result_logger
  )

  res = bohb.run(n_iterations=n_iterations)

  # store results
  with open(os.path.join(working_dir, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

  # shutdown
  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()