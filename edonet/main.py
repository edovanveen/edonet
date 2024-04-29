try:
    import cupy as cp
except ImportError:
    import numpy as cp

import edonet.functions


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
        cp.random.seed(seed)
        self.input_size = input_size
        self.loss, self.loss_d = edonet.functions.loss.choose(loss)
        self.optimizer = edonet.optimizers.choose(self, None)
        
        # Helper function for layer creation.
        def make_layer(layer, inputs, index):
            if layer['type'] == 'Conv2D':
                return edonet.layers.Conv2DLayer(inputs, index, layer['nr_filters'], 
                                                 layer['filter_size'], layer['activation'], 
                                                 layer['stride'], layer['padding'])
            if layer['type'] == 'MaxPool2D':
                return edonet.layers.MaxPool2DLayer(inputs, index, layer['pool_size'])
            if layer['type'] == 'Flatten':
                return edonet.layers.FlattenLayer(inputs, index)
            if layer['type'] == 'Dense':
                return edonet.layers.DenseLayer(inputs, index, layer['nr_nodes'], 
                                                layer['activation'])
            if layer['type'] == 'Dropout':
                return edonet.layers.DropoutLayer(inputs, index, layer['dropout_rate'])
            
        # Make layers.
        self.layers = [make_layer(layers[0], input_size, 0)]
        layer_indices = [i for i in range(len(layers))]
        for i in layer_indices[1:]:
            self.layers.append(make_layer(layers[i], self.layers[i-1].output_size, i))

    def describe(self):
        """
        Describe NeuralNet layout.
        """
        for layer in self.layers:
            print(layer.index, layer.layer_type)
            print("-- input size: ", layer.input_size)
            print("-- output size: ", layer.output_size)
            
    def _predict(self, x, remove_dropout=False):
        """
        Predict labels y from input features x.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples, number of features)
            Input features.
        remove_dropout : bool, optional
            Set dropout values to zero. Default: False.

        Returns
        -------
        y : cp.array of floats, shape (number of examples, number of classes)
            Classification probabilities.
        """
        
        # Push x through each layer using forward propagation.
        y = self.layers[0].forward_prop(x)
        for layer in self.layers[1:]:
            if not remove_dropout or layer.layer_type != 'Dropout':
                y = layer.forward_prop(y)
        return y

    def predict(self, x, batch_size=100, remove_dropout=True):
        """
        Predict labels y from input features x in batches.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples, number of features)
            Input features.
        batch_size : int, optional
            Batch size. Default: 100.
        remove_dropout : bool, optional
            Set dropout values to zero. Default: True.

        Returns
        -------
        y : cp.array of floats, shape (number of examples, number of classes)
            Classification probabilities.
        """
        
        # Calculate number of batches.
        nr_examples = x.shape[0]
        nr_batches = int(-(-nr_examples // batch_size))
        
        # Push x through each layer using forward propagation.
        y = cp.zeros((nr_examples, self.layers[-1].output_size), dtype=cp.float32)
        for _ in self.layers:
            # Iterate over batches.
            for i in range(nr_batches):
                i_start = i*batch_size
                i_end = min(nr_examples, (i+1)*batch_size)
                x_batch = x[i_start:i_end, :]
                y[i_start:i_end, :] = self._predict(x_batch, remove_dropout)
                
        return y

    def evaluate(self, x, y, batch_size=100):
        """
        Evaluate network performance, print accuracy.

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples, number of features)
            Input features.
        y : cp.array
            One-hot encoded labels.
        batch_size : int, optional
            Batch size. Default: 100.
        """
    
        y_pred = self.predict(x, batch_size)
        acc = edonet.metrics.accuracy(y, y_pred)
        print("Evaluation accuracy: ", acc)
        return acc
        
    def grad_desc(self, y_pred, y_true, learning_rate):
        """
        Perform backpropagation and gradient descent.

        Parameters
        ----------
        y_pred : cp.array of floats, shape (number of examples, number of classes)
            One-hot encoded predicted labels.
        y_true : cp.array of floats, shape (number of examples, number of classes)
            One-hot encoded true labels.
        learning_rate : float
            Learning rate for gradient descent.
        """
        
        # Calculate derivative of loss with respect to output of last layer.
        dloss_dx = self.loss_d(y_pred, y_true)
        
        # Do backpropagation and gradient descent, starting at last layer, moving backwards.
        for layer in self.layers[::-1]:
            dloss_dx = layer.back_prop(dloss_dx)
            if layer.has_weights:
                self.optimizer.update(layer, learning_rate)
    
    def fit(self, x, y_true, epochs=1, learning_rate=0.001, batch_size=100, optimizer='same', verbose=False):
        """
        Fit neural net to training set x, y_true

        Parameters
        ----------
        x : cp.array of floats, shape (number of examples, number of features)
            Input features.
        y_true : cp.array of floats, shape (number of examples, number of classes)
            One-hot encoded labels.
        epochs : int, optional
            Number of epochs.
        learning_rate : float, optional
            Learning rate of the network. Default: 0.001.
        batch_size : int, optional
            Batch size. Default: 100.
        optimizer : str or None, optional
            Optimizer identifier. If you want to keep the existing optimizer, set to 'same'. Default: 'same'.
        verbose : bool, optional
            Print update for each batch. Default: False.
        """
        
        # Set optimizer.
        if optimizer != 'same':
            self.optimizer = edonet.optimizers.choose(self, optimizer)
        
        # Calculate number of batches.
        nr_examples = x.shape[0]
        nr_batches = int(-(-nr_examples // batch_size))
            
        # Iterate over epochs.
        for epoch in range(epochs):
            
            print("Epoch: ", epoch)
            avg_loss = cp.zeros(nr_batches)
            avg_acc = cp.zeros(nr_batches)
            
            # Iterate over batches.
            for i in range(nr_batches):
                
                # Forward propagation.
                x_batch = x[i*batch_size:min(nr_examples, (i+1)*batch_size):, :]
                y_batch = y_true[i*batch_size:min(nr_examples, (i+1)*batch_size):, :]
                y_pred = self._predict(x_batch, remove_dropout=False)
                
                # Calculate average loss.
                avg_loss[i] = cp.average(self.loss(y_pred, y_batch))
                avg_acc[i] = edonet.metrics.accuracy(y_batch, y_pred)
                
                # Print status.
                if verbose:
                    print("- Batch: %i/%i, loss: %.3f, acc: %.3f" % (i, nr_batches, avg_loss[i], avg_acc[i]))
                
                # Backpropagation and gradient descent.
                self.grad_desc(y_pred, y_batch, learning_rate)
                
            print("- Average loss: ", cp.average(avg_loss))
