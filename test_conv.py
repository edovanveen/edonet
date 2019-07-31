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
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# Check accuracy.
def accuracy(y_true, y_pred):
    labels_true = y_true.argmax(axis=1)
    labels_pred = y_pred.argmax(axis=1)
    n_good = np.sum(labels_true - labels_pred == 0)
    return n_good / len(y_true)


# Make and test model.
def main():

    """
    np.random.seed(1)
    x = np.random.rand(1, 64, 64, 3)
    
    conv2d = edonet.Conv2DLayer(input_size=(64, 64, 3), index=0, nr_filters=2, 
                                filter_size=(4, 3), activation='relu', stride=(1, 1), padding='same')
    
    import time
    from scipy.signal import convolve, convolve2d
    
    print("---tensordot")
    
    t0 = time.time()
    for i in range(1):
        f = conv2d.forward_prop(x)[0, :, :, :]
    t1 = time.time()
    # print(f)
    print((t1 - t0))
    
    t0 = time.time()
    for i in range(1):
        f = conv2d.back_prop(x[:, :, :, 0:2])[0, :, :, :]
    t1 = time.time()
    print(f[:, :, 0])
    print(conv2d.dloss_dw[0, :, :, :])
    dloss_dz = conv2d.ac_func_d(conv2d.z_cache, x[:, :, :, 0:2])[0, :, :, :]
    print((t1 - t0))
    
    print("---scipy")
    
    f = np.zeros(conv2d.output_size)
    t0 = time.time()
    for i in range(1):
        for c_out in range(2):
            for c_in in range(3):
                f[:, :, c_out] = f[:, :, c_out] + convolve2d(x[0, :, :, c_in], \
                    conv2d.weights[c_in, ::-1, ::-1, c_out], mode='same')
        f = f + conv2d.bias
    t1 = time.time()
    # print(f)
    print((t1 - t0))
    
    f = np.zeros((67, 66, 3))
    t0 = time.time()
    for i in range(1):
        for c_out in range(2):
            for c_in in range(3):        
                f[:, :, c_in] = f[:, :, c_in] + \
                    convolve2d(dloss_dz[:, :, c_out], conv2d.weights[c_in, :, :, c_out], mode='full')
    t1 = time.time()
    print(f[2:-1, 1:-1, 0])
    print((t1 - t0))
    
    f = np.zeros((3, 64, 64, 2))
    t0 = time.time()
    for i in range(1):
        for c_out in range(2):
            for c_in in range(3):
                f[c_in, :, :, c_out] = f[c_in, :, :, c_out] + \
                    convolve(dloss_dz[::-1, ::-1, c_out], x[0, :, :, c_in], mode='same')
    t1 = time.time()
    print(f[0, 30:-30, 31:-30, :])
    print((t1 - t0))
    """
    
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
    model.fit(x_train, y_train, epochs=5, learning_rate=0.001, 
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
