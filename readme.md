## Overview
This is a simple example for an audio classifier using [mfcc](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html) features and a simple cnn architecture using Keras.<br/>
The Dataset is obtained from kaggle and is interfaced using the [Tensorflow Dataset Api](https://www.tensorflow.org/guide/datasets).<br/>
Hyperparameters are optimized using the [HpBandSter](https://github.com/automl/HpBandSter) framework.

## Dataset
Download and extract zip from here into the "dataset/" directory:<br/>
https://www.kaggle.com/mmoreaux/audio-cats-and-dogs/downloads/audio-cats-and-dogs.zip

## Installation
```
pip install -r requirements.txt
```

## Training
```
python train.py
```

## Prediction
```
python predict.py "path-to-wav-file"
```

## Hyperparameter Optimization
```
python train_bohb.py
```
### Why to choose BOHB for hyperparameter optimization
"Modern deep learning methods are very sensitive to many hyperparameters, and, due to the long training times of state-of-the-art models, vanilla Bayesian hyperparameter optimization is typically computationally infeasible. On the other hand, bandit-based configuration evaluation approaches based on random search lack guidance and do not converge to the best configurations as quickly. Here, we propose to combine the benefits of both Bayesian optimization and bandit-based methods, in order to achieve the best of both worlds: strong anytime performance and fast convergence to optimal configurations. We propose a new practical state-of-the-art hyperparameter optimization method, which consistently outperforms both Bayesian optimization and Hyperband on a wide range of problem types, including high-dimensional toy functions, support vector machines, feed-forward neural networks, Bayesian neural networks, deep reinforcement learning, and convolutional neural networks. Our method is robust and versatile, while at the same time being conceptually simple and easy to implement."
<br/>
Stefan Falkner, Aaron Klein, Frank Hutter ; Proceedings of the 35th International Conference on Machine Learning, PMLR 80:1437-1446, 2018.
<br/>
http://proceedings.mlr.press/v80/falkner18a.html
