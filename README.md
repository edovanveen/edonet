# edonet
A minimal neural network for educational purposes, written in Python 3 using only the numpy library for speedy tensor operations.

## Requirements
* numpy

## Usage
To run an example, use `test_dense.py`. For dataset generation and visualisation in `test_dense.py` you will need matplotlib and sklearn.

```python
# Make dataset; the y datasets must be one-hot encoded.
x_train, x_test, y_train, y_test = make_dataset()

# Make and train model using four dense layers and cross-entropy loss.
model = edonet.NeuralNet(input_size=2,
                         layers=({'type': 'dense', 'nr_nodes': 16, 'activation': 'relu'},
                                 {'type': 'dense', 'nr_nodes': 16, 'activation': 'tanh'},
                                 {'type': 'dense', 'nr_nodes': 16, 'activation': 'relu'},
                                 {'type': 'dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                         loss='CEL',
                         seed=0)
                         
# Fit the model to the training data, using 50 iterations.
model.fit(x_train, y_train, epochs=50, learning_rate=0.1, batch_size=10)

# Do a prediction using the test set.
y_pred = model.predict(x_test)
```
## To do

* Figure out correct weight initialization to prevent diminishing/exploding gradients 
  ([link1](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), [link2](https://arxiv.org/pdf/1502.01852.pdf));
* Double check all backprop tensor equations, do 
  [gradient checking](http://cs231n.github.io/neural-networks-3/?source=post_page---------------------------#gradcheck);
* Rethink NeuralNet layers argument, dict currently a bit clunky;
* Add NeuralNet.save() and NeuralNet.load() functionalities;
* Add more activation and loss functions.

## Issues

* CEL loss vectorization goes wrong for certain values of y_pred;
* Inputs should be rescaled properly;
* MaxPool2DLayer: creation of the cached mask goes wrong when multiple elements are equal to max.

## Authors
* **Edo van Veen** - *Initial work* - [edovanveen](https://github.com/edovanveen)
