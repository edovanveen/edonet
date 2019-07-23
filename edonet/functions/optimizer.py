import cupy as cp
import warnings


class NoOptimizer:

    def __init__(self):
        """
        Initialize no optimizer object.
        """
        pass
        
    def update(self, layer, learning_rate):
        """
        Perform vanilla gradient descent.
        
        Parameters
        ----------
        layer : layer object
            Layer within neural network.
        learning_rate : float
            Learning rate.
        """
        layer.update_weights(learning_rate)


# See https://arxiv.org/pdf/1412.6980.pdf
class AdamOptimizer:

    def __init__(self, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer object.
        
        Parameters
        ----------
        model : edonet.NeuralNet object
            Neural network.
        beta1 : float, optional
            Exponential decay rate for first moment. Default: 0.9.
        beta2 : float, optional
            Exponential decay rate for second moment. Default: 0.999.
        epsilon : float, optional
            Small number to prevent division by zero. Default: 1e-8.
        """
        
        # Set attributes.
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time = dict()
        self.moments1_w = dict()
        self.moments1_b = dict()
        self.moments2_w = dict()
        self.moments2_b = dict()
        for layer in model.layers:
            if layer.has_weights:
                self.time[layer.index] = 0
                self.moments1_w[layer.index] = cp.zeros(layer.weights.shape, dtype=cp.float32)
                self.moments2_w[layer.index] = cp.zeros(layer.weights.shape, dtype=cp.float32)
                self.moments1_b[layer.index] = cp.zeros(layer.bias.shape, dtype=cp.float32)
                self.moments2_b[layer.index] = cp.zeros(layer.bias.shape, dtype=cp.float32)
    
    def update(self, layer, learning_rate):
        """
        Perform Adam optimized gradient descent.
        
        Parameters
        ----------
        layer : layer object
            Layer within neural network.
        learning_rate : float
            Learning rate.
        """
        
        # Get index and time.
        index = layer.index
        self.time[index] = self.time[index] + 1
        
        # Update moments.
        self.moments1_w[index] = self.beta1 * self.moments1_w[index] + \
                                 (1 - self.beta1) * layer.dloss_dw
        self.moments1_b[index] = self.beta1 * self.moments1_b[index] + \
                                 (1 - self.beta1) * layer.dloss_db
        self.moments2_w[index] = self.beta2 * self.moments2_w[index] + \
                                 (1 - self.beta2) * cp.power(layer.dloss_dw, 2)
        self.moments2_b[index] = self.beta2 * self.moments2_b[index] + \
                                 (1 - self.beta2) * cp.power(layer.dloss_db, 2)
        update1_w = self.moments1_w[index] / (1 - self.beta1 ** self.time[index])
        update1_b = self.moments1_b[index] / (1 - self.beta1 ** self.time[index])
        update2_w = cp.sqrt(self.moments2_w[index] / (1 - self.beta2 ** self.time[index])) + \
                    self.epsilon
        update2_b = cp.sqrt(self.moments2_b[index] / (1 - self.beta2 ** self.time[index])) + \
                    self.epsilon
                    
        # Set layer weights.
        layer.weights = layer.weights - learning_rate * cp.divide(update1_w, update2_w)
        layer.bias = layer.bias - learning_rate * cp.divide(update1_b, update2_b)
        

def choose(model, optimizer):
    """
    Choose optimizer.
    
    Parameters
    ----------
    model : edonet.NeuralNet object
        Neural network object.
    optimizer : str or None
        String denoting optimizer. For now, either None or 'Adam'.
    
    Returns
    -------
    opt : optimizer object
        Optimizer object with update() method for updating weights.
    """
        
    if optimizer == None:
        opt = NoOptimizer()
    elif optimizer == 'Adam':
        opt = AdamOptimizer(model)
    else:
        warnings.warn("Warning: optimizer '" + str(optimizer) +
                      "' not recognized.")
        opt = NoOptimizer
            
    return opt
