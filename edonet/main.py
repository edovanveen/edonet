import numpy as np
import edonet.functions


class Conv2DLayer:

    def __init__(self, input_size, filter_size, activation, stride=1, padding='valid'):
        return
            
    def init_weights(self):
        return

    def forward_prop(self, x):
        return

    def back_prop(self, dloss_do, learning_rate):
        return
    
    
class MaxPoolLayer:

    def __init__(self, input_size, pool_size, stride=1):
        return

    def forward_prop(self, x):
        return

    def back_prop(self, dloss_do):
        return
    
    
class FlattenLayer:

    def __init__(self, input_size):
        return

    def forward_prop(self, x):
        return

    def back_prop(self, dloss_do):
        return
    
    
class DenseLayer:

    def __init__(self, nr_inputs, nr_nodes, activation):
        """
        Initialize a layer in a neural network.

        Parameters
        ----------
        nr_inputs : int
            Number of input values.
        nr_nodes : int
            Number of nodes.
        activation : str
            For now, either 'relu', 'tanh' or 'softmax'.
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
        Do forward propagation through the network; save x and z to cache.

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
