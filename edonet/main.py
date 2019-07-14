import numpy as np

import edonet.functions


class Conv2DLayer:

    def __init__(self, input_size, nr_filters, filter_size, activation, stride=(1, 1), padding='valid'):
        """
        Initialize a 2D convolution layer.
        
        Parameters
        ----------
        input_size : 3-tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        """
        
        self.input_size = input_size
        self.nr_filters = nr_filters
        self.filter_size = filter_size
        self.stride = stride
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        if padding == 'same':
            self.padding = ((0, 0),
                            (int(np.floor((filter_size[0] - 1) / 2)),
                             int(np.ceil((filter_size[0] - 1) / 2))),
                            (int(np.floor((filter_size[1] - 1) / 2)),
                             int(np.ceil((filter_size[1] - 1) / 2))),
                            (0,0))
        elif padding == 'valid':
            self.padding = ((0, 0), (0, 0), (0, 0) (0, 0))
        self.padded_size = (self.input_size[0] + self.padding[1][0] + self.padding[1][1],
                            self.input_size[1] + self.padding[2][0] + self.padding[2][1],
                            self.input_size[2])
        self.output_size = (int(np.floor((self.padded_size[0] - self.filter_size[0] + 1) / self.stride[0])),
                            int(np.floor((self.padded_size[1] - self.filter_size[1] + 1) / self.stride[1])),
                            self.nr_filters)
            
    def init_weights(self):
        
        
        # Keep track of dimensions
        a, b, c, d = self.filter_size + (self.input_size[2], self.nr_filters)
        
        # Set weight scaling parameter. Can this be improved?
        scaling = np.sqrt(2) / np.sqrt(np.prod(self.filter_size) * np.prod(self.input_size))
        
        # Initialize weights.
        self.filters = scaling * (2 * np.random.rand(a, b, c, d) - 1)
        self.bias = scaling * (2 * np.random.rand(1) - 1)

    def forward_prop(self, x):
        
        # Add padding.
        y = np.pad(x, self.padding, 'constant', constant_values=0)
        
        # Keep track of all the dimensions
        nr_examples, _, _, c = y.shape
        h, i, j, k = y.strides
        p, q = self.filter_size
        n, m, _ = self.output_size
        sx, sy = self.stride
        
        # Create strided submatrices.
        sub_shape = (nr_examples, p, q, n, m, c)
        sub_strides = (h, i * sx, j * sy, i * sx, j * sy, k)
        as_strided = np.lib.stride_tricks.as_strided
        sub_matrices = as_strided(y, shape=sub_shape, strides=sub_strides)
        
        # Get convolution.
        return np.einsum('hijklm,ijmn->hkln', sub_matrices, self.filters)

    def back_prop(self, dloss_do, learning_rate):
        return
    
    
class MaxPool2DLayer:

    def __init__(self, input_size, pool_size=(2, 2)):
        """
        Initialize a 2D max pooling layer.
        
        Parameters
        ----------
        input_size : 3-tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        pool_size : 2-tuple of ints
            Size of the pooling filter. The first two dimensions of the input
            need to be divisible by the pool size dimensions, respectively.
        """
        
        # Save relevant attributes.
        self.input_size = input_size
        self.pool_size = pool_size
        self.output_size = (self.input_size[0] // self.pool_size[0],
                            self.input_size[1] // self.pool_size[1],
                            self.input_size[2])

    def forward_prop(self, x):
        """
        Do forward propagation through the layer, saving an argmax mask to cache.

        Parameters
        ----------
        x : np.array of floats, shape (number of examples,) + self.input_size
            Input values.

        Returns
        -------
        np.array of floats, shape (number of examples,) + self.output_size
           Output values.
        """
        
        # Keep track of all the dimensions.
        nr_examples = x.shape[0]
        m, n, nr_channels = self.output_size
        p, q = self.pool_size
        
        # Reshape x to a tensor from which we can take the maximum of two axes.
        x_reshaped = x.reshape(nr_examples, m, p, n, q, nr_channels)
        
        # Take maximum but keep dimensions.
        y = x_reshaped.max(axis=(2, 4), keepdims=True)
        
        # Make mask of maximum values. Warning: this only works if maximum values are unique!
        # Maybe divide by some kind of count? Would that help?
        self.i_cache = (x_reshaped == y).reshape(x.shape).astype(int)
        
        # Reshape y to new dimensions.
        return y.reshape(nr_examples, m, n, nr_channels)

    def back_prop(self, dloss_do):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_do : np.array of floats, shape (number of examples,) + self.output_size
            Derivative of loss with respect to output values.
            
        Returns
        -------
        np.array of floats, shape (number of examples,) + self.input_size
            Derivative of loss with respect to input values.
        """
        
        # Keep track of all the dimensions.
        nr_examples = dloss_do.shape[0]
        a, b, _ = self.input_size
        m, n, nr_channels = self.output_size
        p, q = self.pool_size
        
        # Expand the derivative to the input shape.
        dloss_do_reshaped = dloss_do.reshape(nr_examples, m, 1, n, 1, nr_channels)
        dloss_do_expanded = np.einsum('abcefh,cd,fg->abdegh', dloss_do_reshaped, np.ones((1, q)), np.ones((1, p)))
        dloss_do_expanded = dloss_do_expanded.reshape(nr_examples, a, b, nr_channels)
        
        # Apply the cached mask to the derivative.
        return np.multiply(dloss_do_expanded, self.i_cache)
    
    
class FlattenLayer:

    def __init__(self, input_size):
        """
        Initialize a flattening layer.
        
        Parameters
        ----------
        input_size : tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        """
        
        # Save relevant attributes.
        self.input_size = input_size
        self.nr_flat = np.prod(input_size)

    def forward_prop(self, x):
        """
        Do forward propagation through the layer.

        Parameters
        ----------
        x : np.array of floats, shape (number of examples,) + self.input_size
            Input values.

        Returns
        -------
        y : np.array of floats, shape (number of examples, self.nr_flat)
           Output values.
        """
        
        # Flatten x (except for first dimension) using reshape.
        return np.reshape(x, newshape=(x.shape[0], self.nr_flat))

    def back_prop(self, dloss_do):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_do : np.array of floats, shape (number of examples, self.nr_flat)
            Derivative of loss with respect to output values.
            
        Returns
        -------
        np.array of floats, shape (number of examples,) + self.input_size
            Derivative of loss with respect to input values.
        """
        
        # Reshape flattened array back to size of input array.
        return np.reshape(dloss_do, newshape=(dloss_do.shape[0],) + self.input_size)
    
    
class DenseLayer:

    def __init__(self, nr_inputs, nr_nodes, activation):
        """
        Initialize a dense layer.

        Parameters
        ----------
        nr_inputs : int
            Number of input values.
        nr_nodes : int
            Number of nodes.
        activation : str
            Activation function, either 'relu', 'tanh' or 'softmax'.
        """
        
        # Save relevant attributes.
        self.nr_inputs = nr_inputs
        self.nr_nodes = nr_nodes
        self.z_cache = np.zeros((nr_nodes, 1))
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        self.init_weights()
            
    def init_weights(self):
        """Initialize weights and bias."""
        
        # Can we optimize this initialization further?
        scaling = np.sqrt(2) / np.sqrt(self.nr_inputs)
        self.weights = scaling * (2 * np.random.rand(self.nr_inputs, self.nr_nodes) - 1)
        self.bias = scaling * (2 * np.random.rand(1) - 1)

    def forward_prop(self, x):
        """
        Do forward propagation through the layer; save x and z to cache.

        Parameters
        ----------
        x : np.array of floats, shape (number of examples, number of input values)
            Input values.

        Returns
        -------
        y : np.array of floats, shape (number of examples, number of nodes)
           Output values.
        """
        
        # Store inputs and weighted inputs in cache.
        self.x_cache = x.copy()
        self.z_cache = np.dot(self.x_cache, self.weights) + self.bias
        
        # Apply activation function.
        y = self.ac_func(self.z_cache)
        return y

    def back_prop(self, dloss_do, learning_rate):
        """
        Do backward propagation through the network and update the weights accordingly.

        Parameters
        ----------
        dloss_do : np.array of floats, shape (number of examples, number of nodes)
            Derivative of loss with respect to output values.
        learning_rate : float
            Learning rate for updating weights.
            
        Returns
        -------
        dloss_dx : np.array of floats, shape (number of examples, number of input values)
            Derivative of loss with respect to input values.
        """
        
        # Calculate derivatives.
        dloss_dz = np.einsum('ab,abc->ac', dloss_do, self.ac_func_d(self.z_cache))
        dloss_dw = np.einsum('ad,ab,cd->abc', dloss_dz, self.x_cache, np.eye(self.nr_nodes))
        dloss_dx = np.einsum('ac,bc->ab', dloss_dz, self.weights)
        
        # Update weights.
        w_update = np.average(dloss_dw, axis=0)
        self.weights = self.weights - learning_rate * w_update
        
        # Return derivative of loss with respect to inputs x
        return dloss_dx


class NeuralNet:

    def __init__(self, input_size, layers, loss='CEL', seed=None):
        """
        Initialize neural network.

        Parameters
        ----------
        input_size : int or tuple of ints
            Number of input features. Should be an int if the first layer is a dense layer.
            Should be a tuple of ints if the first layer is a convolution.
        layers : tuple of dicts
            Tuple of dicts containing layer properties. Example dict:
            {'type': 'dense', 'nr_nodes': 16, 'activation': 'relu'}
        loss : str, optional
            Loss function. Default: 'CEL' for cross-entropy loss.
        seed : None or int, optional
            Seed for random initialization of weights. Default: None.
        """
        
        # Process input.
        np.random.seed(seed)
        self.input_size = input_size
        self.loss, self.loss_d = edonet.functions.loss.choose(loss)
        
        # Helper function for layer creation.
        def make_layer(layer, inputs):
            if layer['type'] == 'dense':
                return DenseLayer(inputs, layer['nr_nodes'], layer['activation'])
            
        # Make layers.
        self.layers = [make_layer(layers[0], input_size)]
        layer_indices = np.arange(len(layers))
        for i in layer_indices[1:]:
            if layers[i-1]['type'] == 'dense':
                nr_inputs = layers[i-1]['nr_nodes']
            self.layers.append(make_layer(layers[i], nr_inputs))

    def predict(self, x):
        """
        Predict labels y from input features x.

        Parameters
        ----------
        x : np.array of floats, shape (number of examples, number of features)
            Input features.

        Returns
        -------
        y : np.array of floats, shape (number of examples, number of classes)
            Classification probabilities.
        """
        
        # Push x through each layer using forward propagation.
        y = x.copy()
        for layer in self.layers:
            y = layer.forward_prop(y)
        return y

    def grad_desc(self, y_pred, y_true, learning_rate):
        """
        Perform backpropagation and gradient descent.

        Parameters
        ----------
        y_pred : np.array of floats, shape (number of examples, number of classes)
            One-hot encoded predicted labels.
        y_true : np.array of floats, shape (number of examples, number of classes)
            One-hot encoded true labels.
        learning_rate : float
            Learning rate for gradient descent.
        """
        
        # Calculate derivative of loss with respect to output of last layer.
        dloss_dx = self.loss_d(y_pred, y_true)
        
        # Do backpropagation and gradient descent, starting at last layer, moving backwards.
        for layer in self.layers[::-1]:
            dloss_dx = layer.back_prop(dloss_dx, learning_rate)
    
    def fit(self, x, y_true, epochs=1, learning_rate=0.1, batch_size=100):
        """
        Fit neural net to training set x, y_true

        Parameters
        ----------
        x : np.array of floats, shape (number of examples, number of features)
            Input features.
        y_true : np.array of floats, shape (number of examples, number of classes)
            One-hot encoded labels.
        epochs : int, optional
            Number of epochs.
        learning_rate : float, optional
            Learning rate of the network. Default: 0.1.
        batch_size : int, optional
            Batch size. Default: 100.
        """
        
        # Calculate number of batches.
        nr_examples = x.shape[0]
        nr_batches = int(np.ceil(nr_examples / batch_size))
            
        # Iterate over epochs.
        for epoch in range(epochs):
            
            print("Epoch: ", epoch)
            avg_loss = np.zeros(nr_batches)
            
            # Iterate over batches.
            for i in range(nr_batches):
                
                # Forward propagation.
                x_batch = x[i*batch_size:min(nr_examples,(i+1)*batch_size):,:]
                y_batch = y_true[i*batch_size:min(nr_examples,(i+1)*batch_size):,:]
                y_pred = self.predict(x_batch)
                
                # Calculate average loss.
                avg_loss[i] = np.average(self.loss(y_pred, y_batch))
                
                # Backpropagation and gradient descent.
                self.grad_desc(y_pred, y_batch, learning_rate)
                
            print("- Average loss: ", np.average(avg_loss))
