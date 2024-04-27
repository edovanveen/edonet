import cupy as cp


def relu(z):
    """
    Perform relu activation function.

    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        Input values.

    Returns
    -------
    cp.array of floats, shape (number of examples,) + (layer shape)
       Output values.
    """
    return cp.maximum(z, 0)


def relu_d(z, dloss_dy):
    """
    Calculate derivatives with respect to inputs of relu function.
    
    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    dloss_dy : cp.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to outputs of relu function.
    
    Returns
    -------
    cp.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to inputs of relu function.
    """
    
    dy_dz = z > 0
    return cp.multiply(dloss_dy, dy_dz)


def tanh(z):
    """
    Perform tanh activation function.

    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        Input values.

    Returns
    -------
    cp.array of floats, shape (number of examples,) + 2 * (layer shape)
       Output values.
    """
    return cp.tanh(z)


def tanh_d(z, dloss_dy):
    """
    Derivatives with respect to inputs of tanh function.
    
    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    dloss_dy : cp.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to outputs of tanh function.
    
    Returns
    -------
    cp.array of floats, shape (number of examples,) + 2 * (layer shape)
        Derivatives of loss with respect to inputs of tanh function.
    """
    
    dy_dz = 1 - cp.square(cp.tanh(z))
    return cp.multiply(dloss_dy, dy_dz)

    
def sigmoid(z):
    """
    Perform sigmoid activation function.

    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        Input values.

    Returns
    -------
    cp.array of floats, shape (number of examples,) + 2 * (layer shape)
       Output values.
    """
    return 1 / (1 + cp.exp(-z))
    
    
def sigmoid_d(z, dloss_dy):
    """
    Derivatives with respect to inputs of sigmoid function.
    
    Parameters
    ----------
    z : cp.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    dloss_dy : cp.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to outputs of sigmoid function.
    
    Returns
    -------
    cp.array of floats, shape (number of examples,) + 2 * (layer shape)
        Derivatives of loss with respect to inputs of sigmoid function.
    """

    dy_dz = sigmoid(z) * (1 - sigmoid(z))
    return cp.multiply(dloss_dy, dy_dz)
    

def softmax(z):
    """
    Perform softmax activation function, only for 1D layers.

    Parameters
    ----------
    z : cp.array of floats, shape (number of examples, number of nodes)
        Input values.

    Returns
    -------
    cp.array of floats, shape (number of examples, number of nodes)
       Output values.
    """
    
    expz = cp.exp(z)
    return cp.divide(expz, cp.sum(expz, axis=1, keepdims=True))


def softmax_d(z, dloss_dy):
    """
    Derivatives with respect to inputs of softmax function, only for 1D layers.
    
    Parameters
    ----------
    z : cp.array of floats, shape (number of examples, number of nodes)
        z-cache.
    dloss_dy : cp.array of floats, shape (number of examples, number of nodes)
        Derivatives of loss with respect to outputs of relu function.
    
    Returns
    -------
    dloss_dz : cp.array of floats, shape (number of examples, number of nodes)
        Derivatives of loss with respect to inputs of softmax function.
    """
    
    # Prepare.
    y = softmax(z)
    eye = cp.eye(y.shape[1], dtype=cp.int8)
    dloss_dz = cp.zeros(y.shape, dtype=cp.float32)
    
    # Iterate over training examples.
    for n in range(y.shape[0]):
        y_tdot_y = -1 * cp.tensordot(y[n], y[n], axes=0)
        y_d_kron = cp.multiply(y[n].reshape(y.shape[1], 1), eye)
        dy_dz = cp.add(y_tdot_y, y_d_kron)
        dloss_dz[n] = cp.tensordot(dloss_dy[n], dy_dz, axes=1)
        
    return dloss_dz


def choose(activation):
    """
    Choose activation function and derivative.
    
    Parameters
    ----------
    activation : str
        String denoting activation function. For now, either 'relu', 'tanh' or 'softmax'.
    
    Returns
    -------
    ac_func : function
        Activation function.
    ac_func_d : function
        Derivative of activation function.
    """
        
    if activation == 'relu':
        ac_func = relu
        ac_func_d = relu_d
    elif activation == 'tanh':
        ac_func = tanh
        ac_func_d = tanh_d
    elif activation == 'sigmoid':
        ac_func = sigmoid
        ac_func_d = sigmoid_d
    elif activation == 'softmax':
        ac_func = softmax
        ac_func_d = softmax_d
    else:
        raise RuntimeError("Activation function '{activation}' not recognized.")
            
    return ac_func, ac_func_d
