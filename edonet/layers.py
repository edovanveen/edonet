import cupy as cp
from abc import ABC, abstractmethod
import edonet.functions


class Layer(ABC):

    @abstractmethod
    def __init__(self, input_size, index):
        self.input_size = input_size
        self.index = index
        self.output_size = input_size
        self.has_weights = False
        self.layer_type = 'AbstractLayer'

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
    
        self.input_size = input_size
        self.index = index
        self.output_size = input_size
        self.has_weights = False
        self.layer_type = 'Dropout'

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
        self._i_cache = cp.random.choice([0, 1], size=(1, self.input_size), 
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

    def __init__(self, input_size, index, nr_filters, filter_size, activation, stride=(1, 1), padding='valid'):
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
        
        # Save attributes.
        self.input_size = input_size
        self.index = index
        self.nr_filters = nr_filters
        self.filter_size = filter_size
        self.stride = stride
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        self.weights = None
        self.bias = None
        self.x_cache = None
        self.z_cache = None
        self.dloss_dw = None
        self.dloss_db = None
        self.has_weights = True
        self.layer_type = 'Conv2D'
        self.init_weights()
        
        # Take care of padding.
        if padding == 'same':
            self.padding = ((0, 0),
                            (int((filter_size[0] - 1) // 2),
                             int(-(-(filter_size[0] - 1) // 2))),
                            (int((filter_size[1] - 1) // 2),
                             int(-(-(filter_size[1] - 1) // 2))),
                            (0, 0))
        elif padding == 'valid':
            self.padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.padded_size = (self.input_size[0] + self.padding[1][0] + self.padding[1][1],
                            self.input_size[1] + self.padding[2][0] + self.padding[2][1],
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
        self.d_x = cp.zeros((a, p, m), dtype=cp.int8)
        for ia in range(a):
            for ip in range(p):
                for ii in range(m):
                    if sx * ii + ip == ia:
                        self.d_x[ia, ip, ii] = 1
        self.d_y = cp.zeros((b, q, n), dtype=cp.int8)
        for ib in range(b):
            for iq in range(q):
                for ij in range(n):
                    if sy * ij + iq == ib:
                        self.d_y[ib, iq, ij] = 1
        
    def init_weights(self):
        """
        Initialize the filter weights. The scaling factor can probably be improved upon.
        """
        
        # Keep track of dimensions
        a, b, c, d = (self.input_size[2],) + self.filter_size + (self.nr_filters,)
        
        # Set weight scaling parameter.
        stdev = cp.sqrt(2) / cp.sqrt(a * b * c * d)
        
        # Initialize weights.
        self.weights = cp.random.normal(loc=0., scale=stdev, size=(a, b, c, d), dtype=cp.float32)
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
        x_pad = cp.pad(x, self.padding, 'constant', constant_values=0)
    
        # Keep track of dimensions.
        nr_examples, _, _, k = x.shape
        m, n, c = self.output_size
        p, q = self.filter_size
        
        # Create x_cache and z_cache.
        x_pad_times_d_x = cp.tensordot(x_pad, self.d_x, axes=((1,), (0,)))
        self.x_cache = cp.tensordot(x_pad_times_d_x, self.d_y, axes=((1,), (0,)))
        self.z_cache = cp.tensordot(self.x_cache, self.weights, 
                                    axes=((1, 2, 4), (0, 1, 2))) + self.bias

        return self.ac_func(self.z_cache)

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
        dloss_dz = self.ac_func_d(self.z_cache, dloss_dy)
        self.dloss_dw = (1/nr_examples) * cp.tensordot(self.x_cache, dloss_dz, 
                                                       axes=((0, 3, 5), (0, 1, 2)))
        d_y_times_filters = cp.tensordot(self.d_y, self.weights, axes=((1,), (2,)))
        dz_dx = cp.tensordot(self.d_x, d_y_times_filters, axes=((1,), (3,)))
        dloss_dx = cp.tensordot(dloss_dz, dz_dx, axes=((1, 2, 3), (1, 3, 5)))
        dloss_dx = dloss_dx[:, self.padding[1][0]:self.padded_size[0] - self.padding[1][1],
                            self.padding[2][0]:self.padded_size[1] - self.padding[2][1], :]
        self.dloss_db = cp.average(self.dloss_dw, axis=(0, 1, 2))
        
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
        
        # Save relevant attributes.
        self.input_size = input_size
        self.index = index
        self.pool_size = pool_size
        self.i_cache = None
        self.output_size = (self.input_size[0] // self.pool_size[0],
                            self.input_size[1] // self.pool_size[1],
                            self.input_size[2])
        self.has_weights = False
        self.layer_type = 'MaxPool2D'

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
        p, q = self.pool_size
        
        # Reshape x to a tensor from which we can take the maximum of two axes.
        x_reshaped = x.reshape((nr_examples, m, p, n, q, nr_channels))
        
        # Take maximum but keep dimensions.
        y = x_reshaped.max(axis=(2, 4), keepdims=True)
        
        # Make mask of maximum values. Warning: this only works if maximum values are unique!
        # Maybe divide by some kind of count? Would that help?
        self.i_cache = cp.array(x_reshaped == y).reshape(x.shape).astype(int)
        
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
        p, q = self.pool_size
        
        # Expand the derivative to the input shape.
        dloss_dy_reshaped = dloss_dy.reshape((nr_examples, m, 1, n, 1, nr_channels))
        dloss_dy_expanded = cp.multiply(dloss_dy_reshaped, cp.ones((1, 1, p, 1, q, 1), dtype=cp.int8))
        dloss_dy_expanded = dloss_dy_expanded.reshape((nr_examples, a, b, nr_channels))
        
        # Apply the cached mask to the derivative.
        return cp.multiply(dloss_dy_expanded, self.i_cache)
        
    
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
        
        # Save relevant attributes.
        self.input_size = input_size
        self.index = index
        self.output_size = input_size[0] * input_size[1] * input_size[2]
        self.has_weights = False
        self.layer_type = 'Flatten'

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
        
        # Save relevant attributes.
        self.input_size = nr_inputs
        self.index = index
        self.output_size = nr_nodes
        self.ac_func, self.ac_func_d = edonet.functions.activation.choose(activation)
        self.weights = None
        self.bias = None
        self.x_cache = None
        self.z_cache = None
        self.dloss_dw = None
        self.dloss_db = None
        self.has_weights = True
        self.layer_type = 'Dense'
        self.init_weights()
            
    def init_weights(self):
        """Initialize weights and bias."""
        
        # Can we optimize this initialization further?
        stdev = cp.sqrt(2) / cp.sqrt(self.output_size)
        # stdev = cp.sqrt(2) / cp.sqrt(self.input_size * self.output_size)
        self.weights = cp.random.normal(loc=0., scale=stdev, 
                                        size=(self.input_size, self.output_size),
                                        dtype=cp.float32)
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
        self.x_cache = x
        self.z_cache = cp.dot(self.x_cache, self.weights) + self.bias
        
        # Apply activation function.
        y = self.ac_func(self.z_cache)
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
        dloss_dz = self.ac_func_d(self.z_cache, dloss_dy)
        self.dloss_dw = cp.tensordot(self.x_cache, dloss_dz, axes=((0,), (0,))) / nr_examples
        dloss_dx = cp.tensordot(dloss_dz, self.weights, axes=((1,), (1,)))
        self.dloss_db = cp.sum(self.dloss_dw, axis=0, keepdims=True) / self.input_size
        
        # Return derivative of loss with respect to inputs x
        return dloss_dx
