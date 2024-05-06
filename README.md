# edonet
A minimal neural network, written in Python 3 using numpy and (optinally) cupy for speedy tensor operations accelerated by CUDA.

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
                                 {'type': 'Dropout', 'dropout_rate': 0.1},
                                 {'type': 'Dense', 'nr_nodes': 8, 'activation': 'relu'},
                                 {'type': 'Dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                         loss='CEL',
                         seed=0)
# Describe model layout.
model.describe()

# Fit the model to the training data, using 5 iterations.
model.fit(x_train, y_train, epochs=5, optimizer='Adam')

# Evaluate using the test set.
model.evaluate(x_test, y_test)
```

Example files:
* To run an example of a densely connected neural network, use `test_dense.py`. For dataset generation and visualisation you will need numpy, matplotlib and sklearn.
* To run an example of a convolutional neural network, use `test_conv.py`. For dataset generation you will need numpy and tensorflow.

## To do

* Weight initialization should depend on activation function
  ([link1](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), [link2](https://arxiv.org/pdf/1502.01852.pdf)) 
  (now we always use relu-optimized weight initialization).
* Double check all backprop tensor equations, do 
  [gradient checking](http://cs231n.github.io/neural-networks-3/?source=post_page---------------------------#gradcheck)
  (already done for Conv2d layer).
* Rethink NeuralNet layers argument type, dict currently a bit clunky.
* Implement BatchNormLayer.
* Add NeuralNet.save() and NeuralNet.load() functionalities for saving weights and structure.
* MaxPool2DLayer: can creation of the cached mask go wrong when multiple elements are equal to max?
* Softmax derivative is slow.
* Conv2D dloss_dx is slow. Write user-defined kernel for this?

## Authors
* **Edo van Veen** - *Initial work* - [edovanveen](https://github.com/edovanveen)
