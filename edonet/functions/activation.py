import numpy as np


def relu(z):
    """
    Perform relu activation function.

    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        Input values.

    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes)
       Output values.
    """
    return np.maximum(z, 0)


def relu_d(z):
    """
    Derivatives with respect to inputs of relu function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        z-cache.
    
    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes, number of nodes)
        Derivatives per example.
    """
    return np.einsum('...b,bc->...bc', np.array(z > 0, dtype=float), np.eye(z.shape[1]))


def tanh(z):
    """
    Perform tanh activation function.

    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        Input values.

    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes)
       Output values.
    """
    return np.tanh(z)


def tanh_d(z):
    """
    Derivatives with respect to inputs of tanh function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        z-cache.
    
    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes, number of nodes)
        Derivatives per example.
    """
    return np.einsum('...b,bc->...bc', 1 - np.square(np.tanh(z)), np.eye(z.shape[1]))


def softmax(z):
    """
    Perform softmax activation function.

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


def softmax_d(z):
    """
    Derivatives with respect to inputs of softmax function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples, number of nodes)
        z-cache.
    
    Returns
    -------
    np.array of floats, shape (number of examples, number of nodes, number of nodes)
        Derivatives per example.
    """
    y = softmax(z)
    return -1. * np.einsum('...b,...c->...bc', y, y) + np.einsum('...b,bc->...bc', y, np.eye(z.shape[1]))


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
    if activation == 'tanh':
        ac_func = tanh
        ac_func_d = tanh_d
    elif activation == 'softmax':
        ac_func = softmax
        ac_func_d = softmax_d
            
    return ac_func, ac_func_d