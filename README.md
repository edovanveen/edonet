# edonet
A simple neural network for educational purposes, written in Python 3, using only numpy.

## Requirements
* numpy

## Usage
Run `test.py`. For dataset generation and visualisation in `test.py`, you need matplotlib and sklearn.

```
# Make dataset; the y datasets are one-hot encoded.
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

## Authors
* **Edo van Veen** - *Initial work* - [edovanveen](https://github.com/edovanveen)
