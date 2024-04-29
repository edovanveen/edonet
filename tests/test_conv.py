try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numpy as np
import gzip
import pickle
import edonet


def load_mnist_file():
    f = gzip.open('tests/data/mnist.pkl.gz', 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()
    return data


# Get mnist handwritten numbers dataset.
def make_mnist_dataset(n_skip=1):
    (x_train, y_train), (x_test, y_test) = load_mnist_file()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))[::n_skip]
    x_test = x_test.reshape((10000, 28, 28, 1))[::n_skip]
    encoder = np.eye(10, dtype=np.int8)
    y_train = [encoder[i] for i in y_train]
    y_train = y_train[::n_skip]
    y_test = [encoder[i] for i in y_test]
    y_test = y_test[::n_skip]
    return cp.array(x_train), cp.array(x_test), cp.array(y_train), cp.array(y_test)


# Make and test model.
def test_mnist():
    
    # Make dataset; only take 1/50th of data to speed up test.
    x_train, x_test, y_train, y_test = make_mnist_dataset(50)
    
    # Make model.
    model = edonet.NeuralNet(input_size=(28, 28, 1),
                             layers=({'type': 'Conv2D', 'nr_filters': 64, 'filter_size': (3, 3),
                                      'activation': 'relu', 'stride': (1, 1), 'padding': 'same'},
                                     {'type': 'MaxPool2D', 'pool_size': (2, 2)},
                                     {'type': 'Conv2D', 'nr_filters': 32, 'filter_size': (3, 3),
                                      'activation': 'relu', 'stride': (1, 1), 'padding': 'valid'},
                                     {'type': 'MaxPool2D', 'pool_size': (2, 2)},
                                     {'type': 'Flatten'},
                                     {'type': 'Dropout', 'dropout_rate': 0.1},
                                     {'type': 'Dense', 'nr_nodes': 32, 'activation': 'relu'},
                                     {'type': 'Dropout', 'dropout_rate': 0.1},
                                     {'type': 'Dense', 'nr_nodes': 10, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)
    model.describe()
              
    # Train model with Adam optimizer.
    model.fit(x_train, y_train, epochs=6, learning_rate=0.001, 
              batch_size=100, optimizer='Adam', verbose=False)
    
    # Check result on test set.
    accuracy = model.evaluate(x_test, y_test)
    assert(accuracy > 0.7)
