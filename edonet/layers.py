from abc import ABC, abstractmethod

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import edonet.functions


class Layer(ABC):

    @abstractmethod
    def __init__(self, input_size, index):
        self.index = index
        self.input_size = input_size
        self.output_size = input_size

        self.has_weights = False
        self.weights = None
        self.bias = None
        self.dloss_dw = None
        self.dloss_db = None

    def forward_prop(self, x):
        pass

    def back_prop(self, dloss_dy):
        pass


class DropoutLayer(Layer):

    def __init__(self, input_size, index, dropout_rate):
        """
        Initialize a dropout layer.
        
        Parameters
        ----------
        input_size : 3-tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        index : int
            Layer index (for identification).
        dropout_rate : float
            Dropout rate.
        """
    
        self.index = index
        self.input_size = input_size
        self.output_size = input_size
        self.has_weights = False

        self._dropout_rate = dropout_rate
        self._keep_rate = 1 - dropout_rate
        self._i_cache = None

    def forward_prop(self, x):
        """
        Do forward propagation through the layer, saving a dropout mask to cache.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples,) + self.input_size
            Input values.

        Returns
        -------
        cp.array of floats, shape (number of examples,) + self.output_size
           Output values.
        """
    
        # Make random dropout mask.
        self._i_cache = cp.random.choice(
            [0, 1],
            size=(1, self.input_size),
            p=[self._dropout_rate, 1 - self._dropout_rate])
        return cp.multiply(x, self._i_cache) / self._keep_rate
    
    def back_prop(self, dloss_dy):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_dy : cp.array of floats, shape (number of examples, self.output_size)
            Derivative of loss with respect to output values.
            
        Returns
        -------
        cp.array of floats, shape (number of examples, self.input_size)
            Derivative of loss with respect to input values.
        """

        return cp.multiply(dloss_dy, self._i_cache) / self._keep_rate


class Conv2DLayer(Layer):

    def __init__(self, input_size, index, nr_filters, filter_size,
                 activation, stride=(1, 1), padding='valid'):
        """
        Initialize a 2D convolution layer.
        
        Parameters
        ----------
        input_size : 3-tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        index : int
            Layer index (for identification).
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

        self.index = index
        self.input_size = input_size
        self.output_size = None  # Set later.

        self.has_weights = True
        self.weights = None
        self.bias = None
        self.dloss_dw = None
        self.dloss_db = None

        self._nr_filters = nr_filters
        self._filter_size = filter_size
        self._stride = stride
        self._ac_func, self._ac_func_d = edonet.functions.activation.choose(activation)
        self._x_cache = None
        self._z_cache = None
        self._init_weights()
        
        # Take care of padding.
        if padding == 'same':
            self._padding = ((0, 0),
                             (int((filter_size[0] - 1) // 2), int(-(-(filter_size[0] - 1) // 2))),
                             (int((filter_size[1] - 1) // 2), int(-(-(filter_size[1] - 1) // 2))),
                             (0, 0))
        elif padding == 'valid':
            self._padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self._padded_size = (self.input_size[0] + self._padding[1][0] + self._padding[1][1],
                             self.input_size[1] + self._padding[2][0] + self._padding[2][1],
                             self.input_size[2])
        
        # Keep track of dimensions.
        a, b, c = self._padded_size
        p, q = self._filter_size
        sx, sy = self._stride
        self.output_size = ((a - p) // sx + 1,
                            (b - q) // sy + 1,
                            self._nr_filters)
        m, n, _ = self.output_size
        
        # Create kronecker delta matrices.
        self._d_x = cp.zeros((a, p, m), dtype=cp.int8)
        for ia in range(a):
            for ip in range(p):
                for ii in range(m):
                    if sx * ii + ip == ia:
                        self._d_x[ia, ip, ii] = 1
        self._d_y = cp.zeros((b, q, n), dtype=cp.int8)
        for ib in range(b):
            for iq in range(q):
                for ij in range(n):
                    if sy * ij + iq == ib:
                        self._d_y[ib, iq, ij] = 1
        
    def _init_weights(self):
        """
        Initialize the filter weights. The scaling factor can probably be improved upon.
        """
        
        # Keep track of dimensions
        a, b, c, d = (self.input_size[2],) + self._filter_size + (self._nr_filters,)
        
        # Set weight scaling parameter.
        stdev = cp.sqrt(2) / cp.sqrt(a * b * c * d)
        
        # Initialize weights.
        self.weights = cp.random.normal(loc=0., scale=stdev, size=(a, b, c, d))
        self.weights = cp.array(self.weights, dtype=cp.float32)
        self.bias = cp.zeros((1, 1, 1, d), dtype=cp.float32)

    def forward_prop(self, x):
        """
        Forward propagation.
        
        Parameters
        ----------
        x : cp.array of floats, shape (nr_examples,) + self.input_size
            Inputs.
            
        Returns
        -------
        cp.array of floats, shape (nr_examples,) + self.output_size
            Outputs.
        """
        
        # Add padding.
        x_pad = cp.pad(x, self._padding, 'constant', constant_values=0)
        
        # Create x_cache and z_cache.
        x_pad_times_d_x = cp.tensordot(x_pad, self._d_x, axes=((1,), (0,)))
        self._x_cache = cp.tensordot(x_pad_times_d_x, self._d_y, axes=((1,), (0,)))
        self._z_cache = cp.tensordot(self._x_cache, self.weights, 
                                     axes=((1, 2, 4), (0, 1, 2))) + self.bias

        return self._ac_func(self._z_cache)

    def back_prop(self, dloss_dy):
        """
        Do backpropagation.
        
        Parameters
        ----------
        dloss_dy : cp.array of floats, shape (nr_examples,) + self.output_size
            Derivative of the loss with respect to output values.
            
        Returns
        -------
        dloss_dx : cp.array of floats, shape (nr_examples,) + self.input_size
            Outputs.
        """
        
        nr_examples = dloss_dy.shape[0]
        
        # Calculate derivatives.
        dloss_dz = self._ac_func_d(self._z_cache, dloss_dy)
        self.dloss_dw = (1/nr_examples) * cp.tensordot(self._x_cache, dloss_dz, 
                                                        axes=((0, 3, 5), (0, 1, 2)))
        self.dloss_db = cp.average(self.dloss_dw, axis=(0, 1, 2))

        d_y_times_filters = cp.tensordot(self._d_y, self.weights, axes=((1,), (2,)))
        dz_dx = cp.tensordot(self._d_x, d_y_times_filters, axes=((1,), (3,)))

        # TODO: this line is very very slow
        dloss_dx = cp.tensordot(dloss_dz, dz_dx, axes=((1, 2, 3), (1, 3, 5)))

        dloss_dx = dloss_dx[:, self._padding[1][0]:self._padded_size[0] - self._padding[1][1],
                            self._padding[2][0]:self._padded_size[1] - self._padding[2][1], :]
        
        # Return derivative of loss with respect to inputs x
        return dloss_dx
    
    
class MaxPool2DLayer(Layer):

    def __init__(self, input_size, index, pool_size=(2, 2)):
        """
        Initialize a 2D max pooling layer.
        
        Parameters
        ----------
        input_size : 3-tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        index : int
            Layer index (for identification).
        pool_size : 2-tuple of ints
            Size of the pooling filter. The first two dimensions of the input
            need to be divisible by the pool size dimensions, respectively.
        """

        self.index = index
        self.input_size = input_size
        self.output_size = (self.input_size[0] // pool_size[0],
                            self.input_size[1] // pool_size[1],
                            self.input_size[2])
        self.has_weights = False

        self._pool_size = pool_size
        self._i_cache = None

    def forward_prop(self, x):
        """
        Do forward propagation through the layer, saving an argmax mask to cache.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples,) + self.input_size
            Input values.

        Returns
        -------
        cp.array of floats, shape (number of examples,) + self.output_size
           Output values.
        """
        
        # Keep track of all the dimensions.
        nr_examples = x.shape[0]
        m, n, nr_channels = self.output_size
        p, q = self._pool_size
        
        # Reshape x to a tensor from which we can take the maximum of two axes.
        x_reshaped = x.reshape((nr_examples, m, p, n, q, nr_channels))
        
        # Take maximum but keep dimensions.
        y = x_reshaped.max(axis=(2, 4), keepdims=True)
        
        # Make mask of maximum values.
        # TODO: this only works if maximum values are unique!
        # Maybe divide by some kind of count? Would that help?
        self._i_cache = cp.array(x_reshaped == y).reshape(x.shape).astype(int)
        
        # Reshape y to new dimensions.
        return y.reshape((nr_examples, m, n, nr_channels))

    def back_prop(self, dloss_dy):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_dy : cp.array of floats, shape (number of examples,) + self.output_size
            Derivative of loss with respect to output values.
            
        Returns
        -------
        cp.array of floats, shape (number of examples,) + self.input_size
            Derivative of loss with respect to input values.
        """
        
        # Keep track of all the dimensions.
        nr_examples = dloss_dy.shape[0]
        a, b, _ = self.input_size
        m, n, nr_channels = self.output_size
        p, q = self._pool_size
        
        # Expand the derivative to the input shape.
        dloss_dy_reshaped = dloss_dy.reshape((nr_examples, m, 1, n, 1, nr_channels))
        dloss_dy_expanded = cp.multiply(dloss_dy_reshaped, cp.ones((1, 1, p, 1, q, 1), dtype=cp.int8))
        dloss_dy_expanded = dloss_dy_expanded.reshape((nr_examples, a, b, nr_channels))
        
        # Apply the cached mask to the derivative.
        return cp.multiply(dloss_dy_expanded, self._i_cache)
        
    
class FlattenLayer(Layer):

    def __init__(self, input_size, index):
        """
        Initialize a flattening layer.
        
        Parameters
        ----------
        input_size : tuple of ints
            Size of the input, e.g. (256, 256, 3) for a 256 by 256 rgb image.
        index : int
            Layer index (for identification).
        """

        self.index = index
        self.input_size = input_size
        self.output_size = input_size[0] * input_size[1] * input_size[2]
        self.has_weights = False

    def forward_prop(self, x):
        """
        Do forward propagation through the layer.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples,) + self.input_size
            Input values.

        Returns
        -------
        y : cp.array of floats, shape (number of examples, self.output_size)
           Output values.
        """
        
        # Flatten x (except for first dimension) using reshape.
        return cp.reshape(x, newshape=(x.shape[0], self.output_size))

    def back_prop(self, dloss_dy):
        """
        Do backward propagation through the layer.

        Parameters
        ----------
        dloss_dy : cp.array of floats, shape (number of examples, self.output_size)
            Derivative of loss with respect to output values.
            
        Returns
        -------
        cp.array of floats, shape (number of examples,) + self.input_size
            Derivative of loss with respect to input values.
        """
        
        # Reshape flattened array back to size of input array.
        return cp.reshape(dloss_dy, newshape=(dloss_dy.shape[0],) + self.input_size)
    
    
class DenseLayer(Layer):

    def __init__(self, nr_inputs, index, nr_nodes, activation):
        """
        Initialize a dense layer.

        Parameters
        ----------
        nr_inputs : int
            Number of input values.
        index : int
            Layer index (for identification).
        nr_nodes : int
            Number of nodes.
        activation : str
            Activation function, either 'relu', 'tanh' or 'softmax'.
        """

        self.index = index
        self.input_size = nr_inputs
        self.output_size = nr_nodes

        self.has_weights = True
        self.weights = None
        self.bias = None
        self.dloss_dw = None
        self.dloss_db = None

        self._ac_func, self._ac_func_d = edonet.functions.activation.choose(activation)
        self._x_cache = None
        self._z_cache = None
        self._init_weights()
            
    def _init_weights(self):
        """Initialize weights and bias."""
        
        # TODO: Can we optimize this initialization further?
        stdev = cp.sqrt(2) / cp.sqrt(self.output_size)
        self.weights = cp.random.normal(loc=0., scale=stdev, 
                                         size=(self.input_size, self.output_size))
        self.weights = cp.array(self.weights, dtype=cp.float32)
        self.bias = cp.zeros((1, self.output_size), dtype=cp.float32)

    def forward_prop(self, x):
        """
        Do forward propagation through the layer; save x and z to cache.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples, number of input values)
            Input values.

        Returns
        -------
        y : cp.array of floats, shape (number of examples, number of nodes)
           Output values.
        """
        
        # Store inputs and weighted inputs in cache.
        self._x_cache = x
        self._z_cache = cp.dot(self._x_cache, self.weights) + self.bias
        
        # Apply activation function.
        y = self._ac_func(self._z_cache)
        return y

    def back_prop(self, dloss_dy):
        """
        Do backward propagation through the network and update the weights accordingly.

        Parameters
        ----------
        dloss_dy : cp.array of floats, shape (number of examples, number of nodes)
            Derivative of loss with respect to output values.
            
        Returns
        -------
        dloss_dx : cp.array of floats, shape (number of examples, number of input values)
            Derivative of loss with respect to input values.
        """
        
        nr_examples = dloss_dy.shape[0]
        
        # Calculate derivatives.
        dloss_dz = self._ac_func_d(self._z_cache, dloss_dy)
        self.dloss_dw = cp.tensordot(self._x_cache, dloss_dz, axes=((0,), (0,))) / nr_examples
        dloss_dx = cp.tensordot(dloss_dz, self.weights, axes=((1,), (1,)))
        self.dloss_db = cp.sum(self.dloss_dw, axis=0, keepdims=True) / self.input_size
        
        # Return derivative of loss with respect to inputs x
        return dloss_dx
