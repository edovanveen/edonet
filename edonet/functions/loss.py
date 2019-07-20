import numpy as np
import warnings


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
    
    epsilon = 1e-16
    return -1. * np.sum(np.multiply(y_true, np.log(np.maximum(y_pred, epsilon))), axis=1)


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
    
    return -1. * np.divide(y_true, y_pred)


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
    else:
        warnings.warn("Warning: loss function '" + str(loss) +
                      "' not recognized, using cross-entropy loss.")
        loss_func = cel
        loss_func_d = cel_d
            
    return loss_func, loss_func_d
