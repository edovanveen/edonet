import cupy as cp
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
    encoder = np.eye(10, dtype=np.int8)
    y_train = [encoder[i] for i in y_train]
    y_test = [encoder[i] for i in y_test]
    return cp.array(x_train), cp.array(x_test), cp.array(y_train), cp.array(y_test)


# Check accuracy.
def accuracy(y_true, y_pred):
    labels_true = y_true.argmax(axis=1)
    labels_pred = y_pred.argmax(axis=1)
    n_good = cp.sum(labels_true - labels_pred == 0)
    return n_good / len(y_true)


# Make and test model.
def main():
    
    # Set memory pool.
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    pinned_memory_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
    
    # Make dataset.
    x_train, x_test, y_train, y_test = make_dataset()
    
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

    # Train model with Adam optimizer.
    model.fit(x_train, y_train, epochs=6, learning_rate=0.001, 
              batch_size=100, optimizer='Adam', verbose=True)
    
    # After training, set dropout rates to 0.
    model.layers[5].dropout_rate = 0
    model.layers[7].dropout_rate = 0
    
    # Show result on test set.
    print("test labels:")
    print(y_test.argmax(axis=1))
    print("pred labels:")
    y_pred = model.batch_predict(x_test, batch_size=200)
    print(y_pred.argmax(axis=1))
    print("accuracy: ", accuracy(y_test, y_pred))


if __name__ == "__main__":
    main()
