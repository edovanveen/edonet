import numpy as np


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


def relu_d(z):
    """
    Derivatives with respect to inputs of relu function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    
    Returns
    -------
    np.array of floats, shape (number of examples,) + 2 * (layer shape)
        Derivatives per example.
    """
    
    # 1D case.
    if len(z.shape) == 2:
        return np.einsum('ab,bc->abc', np.array(z > 0, dtype=int), np.eye(z.shape[1]))
    
    # 2D + channels case.
    elif len(z.shape) == 4:
        return np.array(z > 0, dtype=int)


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


def tanh_d(z):
    """
    Derivatives with respect to inputs of tanh function.
    
    Parameters
    ----------
    z : np.array of floats, shape (number of examples,) + (layer shape)
        z-cache.
    
    Returns
    -------
    np.array of floats, shape (number of examples,) + 2 * (layer shape)
        Derivatives per example.
    """
    
    # 1D case.
    if len(z.shape) == 2:
        return np.einsum('ab,bc->abc', 1 - np.square(np.tanh(z)), np.eye(z.shape[1]))
    
    # 2D + channels case.
    elif len(z.shape) == 4:
        return 1 - np.square(np.tanh(z))


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


def softmax_d(z):
    """
    Derivatives with respect to inputs of softmax function, only for 1D layers.
    
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
    return -1. * np.einsum('ab,ac->abc', y, y) + np.einsum('ab,bc->abc', y, np.eye(z.shape[1]))


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