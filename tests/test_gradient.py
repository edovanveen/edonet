import numpy as np
import cupy as cp
import edonet


# Make and test model.
def test_gradient_values():
    
    # Make input data.
    np.random.seed(0)
    x = cp.array(np.random.rand(1, 2, 2, 2))
    y = cp.array([[0, 1]])
    
    # Make and train model.
    model = edonet.NeuralNet(input_size=(2, 2, 2),
                             layers=({'type': 'Conv2D', 'nr_filters': 2, 'filter_size': (3, 3),
                                      'activation': 'relu', 'stride': (1, 1), 'padding': 'same'},
                                     {'type': 'MaxPool2D', 'pool_size': (2, 2)},
                                     {'type': 'Flatten'},
                                     {'type': 'Dense', 'nr_nodes': 4, 'activation': 'relu'},
                                     {'type': 'Dense', 'nr_nodes': 4, 'activation': 'tanh'},
                                     {'type': 'Dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)     
    
    # Calculate derivative of loss with respect to output of last layer.
    y_pred = model.predict(x)
    dloss_dx = model.loss_d(y_pred, y)
    
    # Do backpropagation and store derivatives.
    for layer in model.layers[::-1]:
        dloss_dx = layer.back_prop(dloss_dx)

    # Check derivatives of loss wrt inputs x.
    epsilon = 1e-3
    dloss_dx_check = np.zeros(dloss_dx.shape)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_check0 = x.copy()
                x_check1 = x.copy()
                x_check0[0, i, j, k] = x[0, i, j, k] - epsilon
                x_check1[0, i, j, k] = x[0, i, j, k] + epsilon
                y_check0 = model.predict(x_check0)
                y_check1 = model.predict(x_check1)
                dloss_dx_check[0, i, j, k] = (model.loss(y_check1, y)[0] - 
                                              model.loss(y_check0, y)[0]) / (2 * epsilon)
    difference = np.max(cp.asnumpy(dloss_dx) - dloss_dx_check)
    assert(difference < 1e-4)
    
    # Check derivatives of loss wrt weights of conv2d layer.
    dloss_dw = model.layers[0].dloss_dw
    orig_filters = model.layers[0].weights.copy()
    dloss_dw_check = np.zeros(dloss_dw.shape)
    for h in range(2):
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    model.layers[0].weights = orig_filters.copy()
                    model.layers[0].weights[h, i, j, k] = orig_filters[h, i, j, k] - epsilon
                    y_check0 = model.predict(x)
                    model.layers[0].weights[h, i, j, k] = orig_filters[h, i, j, k] + epsilon
                    y_check1 = model.predict(x)
                    dloss_dw_check[h, i, j, k] = (model.loss(y_check1, y)[0] - 
                                                  model.loss(y_check0, y)[0]) / (2 * epsilon)
    difference = np.max(cp.asnumpy(dloss_dw) - dloss_dw_check)
    assert(difference < 1e-4)
