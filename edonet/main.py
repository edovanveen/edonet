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
        nr_filters : int
            Number of convolutional filters.
        filter_size : 2-tuple of ints
            Size of the convolutional filters.
        activation : str
            Activation function.
        stride : 2-tuple of ints, optional
            x and y strides for applying the filters. Default: (1, 1).
        padding : str, optional
            Padding - 'valid' or 'same'. Default: 'valid'.
        """
        
        # Save attributes.
        self.input_size = input_size
        self.nr_filters = nr_filters
        self.filter_size = filter_size
        self.stride = stride
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        self.filters = None
        self.bias = None
        self.x_cache = None
        self.z_cache = None
        
        # Take care of padding.
        if padding == 'same':
            self.padding = ((0, 0),
                            (int(np.floor((filter_size[0] - 1) / 2)),
                             int(np.ceil((filter_size[0] - 1) / 2))),
                            (int(np.floor((filter_size[1] - 1) / 2)),
                             int(np.ceil((filter_size[1] - 1) / 2))),
                            (0, 0))
        elif padding == 'valid':
            self.padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.padded_size = (self.input_size[0] + np.sum(self.padding[1]),
                            self.input_size[1] + np.sum(self.padding[2]),
                            self.input_size[2])
        
        # Keep track of dimensions.
        a, b, c = self.padded_size
        p, q = self.filter_size
        sx, sy = self.stride
        self.output_size = ((a - p) // sx + 1,
                            (b - q) // sy + 1,
                            self.nr_filters)
        m, n, _ = self.output_size
        
        # Create kronecker delta matrices.
        self.d_x = np.zeros((a, p, m), dtype=int)
        for ia in range(a):
            for ip in range(p):
                for ii in range(m):
                    if sx * ii + ip == ia:
                        self.d_x[ia, ip, ii] = 1
        self.d_y = np.zeros((b, q, n), dtype=int)
        for ib in range(b):
            for iq in range(q):
                for ij in range(n):
                    if sy * ij + iq == ib:
                        self.d_y[ib, iq, ij] = 1
                        
        # Create intial filter weights.
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize the filter weights. The scaling factor can probably be improved upon.
        """
        
        # Keep track of dimensions
        a, b, c, d = (self.input_size[2],) + self.filter_size + (self.nr_filters,)
        
        # Set weight scaling parameter.
        stdev = np.sqrt(2) / np.sqrt(a * b * c * d)
        
        # Initialize weights.
        self.filters = np.random.normal(loc=0., scale=stdev, size=(a, b, c, d))
        self.bias = np.zeros((1, 1, 1, d))

    def forward_prop(self, x):
        """
        Forward propagation.
        
        Parameters
        ----------
        x : np.array of floats, shape (nr_examples,) + self.input_size
            Inputs.
            
        Returns
        -------
        np.array of floats, shape (nr_examples,) + self.output_size
            Outputs.
        """
        
        # Add padding.
        x_pad = np.pad(x, self.padding, 'constant', constant_values=0)
    
        # Keep track of dimensions.
        nr_examples, _, _, k = x.shape
        m, n, c = self.output_size
        p, q = self.filter_size
        
        # Create x_cache and z_cache.
        x_pad_times_d_x = np.tensordot(x_pad, self.d_x, axes=((1,), (0,)))
        self.x_cache = np.tensordot(x_pad_times_d_x, self.d_y, axes=((1,), (0,)))
        self.z_cache = np.tensordot(self.x_cache, self.filters, 
                                    axes=((1, 2, 4), (0, 1, 2))) + self.bias

        return self.ac_func(self.z_cache)

    def back_prop(self, dloss_dy, learning_rate, update_weights=True, store_derivatives=False):
        """
        Do backpropagation and update weights.
        
        Parameters
        ----------
        dloss_dy : np.array of floats, shape (nr_examples,) + self.output_size
            Derivative of the loss with respect to output values.
        learning_rate : float
            Learning rate.
        update_weights : bool, optional
            Update weights during back propagation. Default: True.
        store_derivatives : bool, optional
            Store the weight and bias derivatives. Default: False
            
        Returns
        -------
        dloss_dx : np.array of floats, shape (nr_examples,) + self.input_size
            Outputs.
        """
        
        nr_examples = dloss_dy.shape[0]
        
        # Calculate derivatives.
        dloss_dz = self.ac_func_d(self.z_cache, dloss_dy)
        dloss_dw = (1/nr_examples) * np.tensordot(self.x_cache, dloss_dz, 
                                                  axes=((0, 3, 5), (0, 1, 2)))
        d_y_times_filters = np.tensordot(self.d_y, self.filters, axes=((1,), (2,)))
        dz_dx = np.tensordot(self.d_x, d_y_times_filters, axes=((1,), (3,)))
        dloss_dx = np.tensordot(dloss_dz, dz_dx, axes=((1, 2, 3), (1, 3, 5)))
        dloss_dx = dloss_dx[:, self.padding[1][0]:self.padded_size[0] - self.padding[1][1],
                            self.padding[2][0]:self.padded_size[1] - self.padding[2][1], :]
        dloss_dw = np.average(dloss_dw, axis=0)
        dloss_db = np.average(dloss_dw, axis=(0, 1))

        # Update weights.
        if update_weights:
            self.filters = self.filters - learning_rate * dloss_dw
            self.bias = self.bias - learning_rate * dloss_db
        
        # Store derivatives.
        if store_derivatives:
            self.dloss_dw = dloss_dw
            self.dloss_db = dloss_db
        
        # Return derivative of loss with respect to inputs x
        return dloss_dx
    
    
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
        self.i_cache = None
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
        x_reshaped = x.reshape((nr_examples, m, p, n, q, nr_channels))
        
        # Take maximum but keep dimensions.
        y = x_reshaped.max(axis=(2, 4), keepdims=True)
        
        # Make mask of maximum values. Warning: this only works if maximum values are unique!
        # Maybe divide by some kind of count? Would that help?
        self.i_cache = np.array(x_reshaped == y).reshape(x.shape).astype(int)
        
        # Reshape y to new dimensions.
        return y.reshape((nr_examples, m, n, nr_channels))

    def back_prop(self, dloss_dy, _0, _1=True, _2=False):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_dy : np.array of floats, shape (number of examples,) + self.output_size
            Derivative of loss with respect to output values.
        _0 : placeholder
            Placeholder parameter for learning_rate.
        _1 : placeholder
            Placeholder parameter for update_weights.
        _2 : placeholder
            Placeholder parameter for store_derivatives.
            
        Returns
        -------
        np.array of floats, shape (number of examples,) + self.input_size
            Derivative of loss with respect to input values.
        """
        
        # Keep track of all the dimensions.
        nr_examples = dloss_dy.shape[0]
        a, b, _ = self.input_size
        m, n, nr_channels = self.output_size
        p, q = self.pool_size
        
        # Expand the derivative to the input shape.
        dloss_dy_reshaped = dloss_dy.reshape((nr_examples, m, 1, n, 1, nr_channels))
        dloss_dy_expanded = np.multiply(dloss_dy_reshaped, np.ones((1, 1, p, 1, q, 1)))
        dloss_dy_expanded = dloss_dy_expanded.reshape((nr_examples, a, b, nr_channels))
        
        # Apply the cached mask to the derivative.
        return np.multiply(dloss_dy_expanded, self.i_cache)
    
    
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
        self.output_size = np.prod(input_size)
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

    def back_prop(self, dloss_do, _0, _1=True, _2=False):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_do : np.array of floats, shape (number of examples, self.nr_flat)
            Derivative of loss with respect to output values.
        _ : placeholder
            Placeholder parameter for learning_rate.
        _1 : placeholder
            Placeholder parameter for update_weights.
        _2 : placeholder
            Placeholder parameter for store_derivatives.
            
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
        self.output_size = nr_nodes
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        self.weights = None
        self.bias = None
        self.x_cache = None
        self.z_cache = None
        self.init_weights()
            
    def init_weights(self):
        """Initialize weights and bias."""
        
        # Can we optimize this initialization further?
        stdev = np.sqrt(2) / np.sqrt(self.output_size)
        # stdev = np.sqrt(2) / np.sqrt(self.nr_inputs * self.output_size)
        self.weights = np.random.normal(loc=0., scale=stdev, size=(self.nr_inputs, self.output_size))
        self.bias = np.zeros((1, self.output_size))

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

    def back_prop(self, dloss_dy, learning_rate, update_weights=True, store_derivatives=False):
        """
        Do backward propagation through the network and update the weights accordingly.

        Parameters
        ----------
        dloss_dy : np.array of floats, shape (number of examples, number of nodes)
            Derivative of loss with respect to output values.
        learning_rate : float
            Learning rate for updating weights.
        update_weights : bool, optional
            Update weights during back propagation. Default: True.
        store_derivatives : bool, optional
            Store the weight and bias derivatives. Default: False
            
        Returns
        -------
        dloss_dx : np.array of floats, shape (number of examples, number of input values)
            Derivative of loss with respect to input values.
        """
        
        nr_examples = dloss_dy.shape[0]
        
        # Calculate derivatives.
        dloss_dz = self.ac_func_d(self.z_cache, dloss_dy)
        dloss_dw = np.tensordot(self.x_cache, dloss_dz, axes=((0,), (0,))) / nr_examples
        dloss_dx = np.tensordot(dloss_dz, self.weights, axes=((1,), (1,)))
        dloss_db = np.sum(dloss_dw, axis=0, keepdims=True) / self.nr_inputs
        
        # Update weights.
        if update_weights:
            self.weights = self.weights - learning_rate * dloss_dw
            self.bias = self.bias - learning_rate * dloss_db
        
        # Store derivatives.
        if store_derivatives:
            self.dloss_dw = dloss_dw
            self.dloss_db = dloss_db
        
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
            if layer['type'] == 'conv2D':
                return Conv2DLayer(inputs, layer['nr_filters'], layer['filter_size'], 
                                   layer['activation'], layer['stride'], layer['padding'])
            if layer['type'] == 'maxpool':
                return MaxPool2DLayer(inputs, layer['pool_size'])
            if layer['type'] == 'flatten':
                return FlattenLayer(inputs)
            if layer['type'] == 'dense':
                return DenseLayer(inputs, layer['nr_nodes'], layer['activation'])
            
        # Make layers.
        self.layers = [make_layer(layers[0], input_size)]
        layer_indices = np.arange(len(layers))
        for i in layer_indices[1:]:
            self.layers.append(make_layer(layers[i], self.layers[i-1].output_size))

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
    
    def fit(self, x, y_true, epochs=1, learning_rate=0.1, batch_size=100, verbose=False):
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
        verbose : bool, optional
            Print update for each batch. Default: False
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
                x_batch = x[i*batch_size:min(nr_examples, (i+1)*batch_size):, :]
                y_batch = y_true[i*batch_size:min(nr_examples, (i+1)*batch_size):, :]
                y_pred = self.predict(x_batch)
                
                # Calculate average loss.
                avg_loss[i] = np.average(self.loss(y_pred, y_batch))
                
                # Print status.
                if verbose:
                    print("- Batch: ", i, "/", nr_batches, ", loss: ", avg_loss[i])
                
                # Backpropagation and gradient descent.
                self.grad_desc(y_pred, y_batch, learning_rate)
                
            print("- Average loss: ", np.average(avg_loss))
