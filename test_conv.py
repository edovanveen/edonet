import numpy as np
import tensorflow as tf
import edonet


# Make test dataset.
def make_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    encoder = np.eye(10, dtype=int)
    y_train = np.array([encoder[i] for i in y_train])
    y_test = np.array([encoder[i] for i in y_test])
    return x_train, x_test, y_train, y_test


# Check accuracy.
def accuracy(y_true, y_pred):
    labels_true = y_true.argmax(axis=1)
    labels_pred = y_pred.argmax(axis=1)
    n_good = np.sum(labels_true - labels_pred == 0)
    return n_good / len(y_true)


# Make and test model.
def main():
    
    # Make dataset.
    x_train, x_test, y_train, y_test = make_dataset()
    
    # Make model.
    model = edonet.NeuralNet(input_size=(28, 28, 1),
                             layers=({'type': 'conv2D', 'nr_filters': 50, 'filter_size': (3, 3),
                                      'activation': 'relu', 'stride': (1, 1), 'padding': 'valid'},
                                     {'type': 'maxpool', 'pool_size': (2, 2)},
                                     {'type': 'flatten'},
                                     {'type': 'dense', 'nr_nodes': 50, 'activation': 'relu'},
                                     {'type': 'dense', 'nr_nodes': 10, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)

    # Train model with decreasing learning rate.
    model.fit(x_train, y_train, epochs=6, learning_rate=0.001, batch_size=200, optimizer='Adam', verbose=True)

    # Show result on test set.
    print("test labels:")
    print(y_test.argmax(axis=1))
    print("pred labels:")
    print(model.predict(x_test).argmax(axis=1))
    print("accuracy: ", accuracy(y_test, model.predict(x_test))) 


if __name__ == "__main__":
    main()
