import numpy as np
import warnings


def relu(z):
    """
    Perform relu activation function.

    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        Input values.

    Returns
    -------
    np.array of floats, shape (number of examples,) + (layer shape)
       Output values.
    """
    return np.maximum(z, 0)


def relu_d(z, dloss_dy):
    """
    Calculate derivatives with respect to inputs of relu function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    dloss_dy : np.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to outputs of relu function.
    
    Returns
    -------
    np.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to inputs of relu function.
    """
    
    dy_dz = np.array(z > 0, dtype=int)
    return np.multiply(dloss_dy, dy_dz)


def tanh(z):
    """
    Perform tanh activation function.

    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        Input values.

    Returns
    -------
    np.array of floats, shape (number of examples,) + 2 * (layer shape)
       Output values.
    """
    return np.tanh(z)


def tanh_d(z, dloss_dy):
    """
    Derivatives with respect to inputs of tanh function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    dloss_dy : np.array of floats, shape (number of examples,) + (layer shape)
        Derivatives of loss with respect to outputs of tanh function.
    
    Returns
    -------
    np.array of floats, shape (number of examples,) + 2 * (layer shape)
        Derivatives of loss with respect to inputs of tanh function.
    """
    
    dy_dz = 1 - np.square(np.tanh(z))
    return np.multiply(dloss_dy, dy_dz)


def softmax(z):
    """
    Perform softmax activation function, only for 1D layers.

    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        Input values.

    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes)
       Output values.
    """
    
    expz = np.subtract(np.exp(z), np.amax(z, axis=1, keepdims=True))
    return np.divide(expz, np.sum(expz, axis=1, keepdims=True))


def softmax_d(z, dloss_dy):
    """
    Derivatives with respect to inputs of softmax function, only for 1D layers.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        z-cache.
    dloss_dy : np.array of floats, shape (number of examples, number of nodes)
        Derivatives of loss with respect to outputs of relu function.
    
    Returns
    -------
    dloss_dz : np.array of floats, shape (number of examples, number of nodes, number of nodes)
        Derivatives of loss with respect to inputs of softmax function.
    """
    
    # Prepare.
    y = softmax(z)
    eye = np.eye(y.shape[1])
    dloss_dz = np.zeros(y.shape)
    
    # Iterate over training examples.
    for n in range(y.shape[0]):
        y_tdot_y = -1 * np.tensordot(y[n], y[n], axes=0)
        y_d_kron = np.multiply(y[n].reshape(y.shape[1], 1), eye)
        dy_dz = np.add(y_tdot_y, y_d_kron)
        dloss_dz[n] = np.tensordot(dloss_dy[n], dy_dz, axes=1)
        
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
    elif activation == 'softmax':
        ac_func = softmax
        ac_func_d = softmax_d
    else:
        warnings.warn("Warning: activation function '" + str(activation) +
                      "' not recognized, using relu activation function.")
        ac_func = relu
        ac_func_d = relu_d
            
    return ac_func, ac_func_d
