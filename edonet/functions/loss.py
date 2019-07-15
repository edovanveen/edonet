import numpy as np


def cel(y_pred, y_true):
    """
    Perform cross-entropy loss function.

    Parameters
    ----------
    y_true : np.array of floats, shape (number of examples, number of classes)
        True labels, one hot encoded.
    y_pred : np.array of floats, shape (number of examples, number of classes)
        Predicted labels, one hot encoded.

    Returns
    -------
    np.array of floats, shape (number of examples)
       Loss values per example.
    """
    
    m, n = y_true.shape
    loss = np.zeros(m)
    for i in range(m):
        for j in range(n):
            y0 = y_true[i, j]
            y1 = y_pred[i, j]
            if y0 > 0:
                if y1 > 0.0001:
                    loss[i] = loss[i] - y0 * np.log(y1)
                else:
                    loss[i] = loss[i] + 10
    return loss
    # The following gives errors when y_pred is too close to zero:
    # return -1. * np.sum(np.multiply(y_true, np.log(y_pred)), axis=1)


def cel_d(y_pred, y_true):
    """
    Get the derivatives with respect to y_pred of the cross-entropy loss function.

    Parameters
    ----------
    y_pred : np.array of floats, shape (number of examples, number of classes)
        Predicted labels, one hot encoded.
    y_true : np.array of floats, shape (number of examples, number of classes)
        True labels, one hot encoded.

    Returns
    -------
    np.array of floats, shape (number of examples, number of classes)
       Derivative of the loss function with respect to y_pred.
    """
    
    m, n = y_true.shape
    loss_d = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            y0 = y_true[i, j]
            y1 = y_pred[i, j]
            if y0 > 0:
                if y1 > 0.001:
                    loss_d[i, j] = - y0 / y1
                else:
                    loss_d[i, j] = -1000
    return loss_d
    
    # The following gives errors when y_pred is too close to zero:
    # return -1. * np.divide(y_true, y_pred)


def choose(loss):
    """
    Choose loss function and derivative.
    
    Parameters
    ----------
    loss : str
        String denoting loss function. For now, only 'CEL'.
    
    Returns
    -------
    loss_func : function
        Loss function.
    loss_func_d : function
        Derivative of loss function.
    """
        
    if loss == 'CEL':
        loss_func = cel
        loss_func_d = cel_d
            
    return loss_func, loss_func_d
