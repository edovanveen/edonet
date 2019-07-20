import numpy as np
import tensorflow as tf
import edonet


# Make and test model.
def main():
    
    # Make input data.
    x = np.random.rand((1, 6, 6, 3))
    
    # Make and train model.
    model = edonet.NeuralNet(input_size=(6, 6, 1),
                             layers=({'type': 'conv2D', 'nr_filters': 2, 'filter_size': (3, 3),
                                      'activation': 'relu', 'stride': (1, 1), 'padding': 'valid'},
                                     {'type': 'maxpool', 'pool_size': (2, 2)},
                                     {'type': 'flatten'},
                                     {'type': 'dense', 'nr_nodes': 4, 'activation': 'relu'},
                                     {'type': 'dense', 'nr_nodes': 4, 'activation': 'tanh'},
                                     {'type': 'dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)     
    
    # Calculate derivative of loss with respect to output of last layer.
    y_pred = self.predict(x)
    dloss_dx = self.loss_d(y_pred, y_true)
    
    # Do backpropagation and store derivatives.
    for layer in self.layers[::-1]:
        dloss_dx = layer.back_prop(dloss_dx, learning_rate, 
                                   update_weights=True, store_derivatives=True)


if __name__ == "__main__":
    main()
