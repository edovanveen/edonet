# edonet
A minimal neural network, written in Python 3 using only the CuPy library for speedy tensor operations accelerated by CUDA.

## Requirements
* CuPy

## Usage

Example code:
```python
# Make dataset; the y datasets must be one-hot encoded.
x_train, x_test, y_train, y_test = make_dataset()

# Make model using a convolutional layer, a maxpool layer, a flatten layer and two dense layers.
# Inputs are 32 * 32 pixel rgb images, outputs are 2 classes. 
# We use relu activation functions and cross-entropy loss.
model = edonet.NeuralNet(input_size=(32, 32, 3),
                         layers=({'type': 'Conv2D', 'nr_filters': 16, 'filter_size': (3, 3),
                                  'activation': 'relu', 'stride': (1, 1), 'padding': 'valid'},
                                 {'type': 'MaxPool2D', 'pool_size': (2, 2)},
                                 {'type': 'Flatten'},
                                 {'type': 'Dense', 'nr_nodes': 16, 'activation': 'relu'},
                                 {'type': 'Dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                         loss='CEL',
                         seed=0)
                         
# Fit the model to the training data, using 5 iterations.
model.fit(x_train, y_train, epochs=5, optimizer='Adam')

# Do a prediction using the test set.
y_pred = model.predict(x_test)
```

Example files:
* To run an example of a densely connected neural network, use `test_dense.py`. For dataset generation and visualisation in `test_dense.py` you will need numpy, matplotlib and sklearn.
* To run an example of a convolutional neural network, use `test_conv.py`. For dataset generation you will need numpy and tensorflow.

## To do

* Research: can we further speed up the CuPy operations? When using convolutions, 
  the GPU seems to be using 'copy' for parts of the backprop computation.
  Maybe write a nice user-defined kernel for calculating dloss_dx?
  Or use FFT convolution? Or use chainer.functions.convolution_2d?
* Weight initialization should depend on activation function
  ([link1](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), [link2](https://arxiv.org/pdf/1502.01852.pdf)) 
  (now we always use relu-optimized weight initialization).
* Double check all backprop tensor equations, do 
  [gradient checking](http://cs231n.github.io/neural-networks-3/?source=post_page---------------------------#gradcheck)
  (already done for Conv2d layer).
* Rethink NeuralNet layers argument type, dict currently a bit clunky.
* Implement NeuralNet.evaluate().
* Implement BatchNormLayer.
* Add NeuralNet.save() and NeuralNet.load() functionalities for saving weights and structure.
* Add NeuralNet.describe() functionality that shows all layers and sizes.
* Automatically set dropout_rate to 0 in DropoutLayers when calling NeuralNet.predict().
* Add more activation and loss functions, as well as optimizers.

## Issues

* MaxPool2DLayer: creation of the cached mask goes wrong when multiple elements are equal to max.
* Softmax derivative is slow.

## Authors
* **Edo van Veen** - *Initial work* - [edovanveen](https://github.com/edovanveen)
