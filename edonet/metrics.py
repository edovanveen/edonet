import cupy as cp


def accuracy(y_pred, y_true):
    """
    Calculate accuracy of one-hot encoded output.
    
    Parameters
    ----------
    y_pred : np.array
        One-hot encoded predicted labels.
    y_true : np.array
        One-hot encoded true labels.
    """
    
    # Calculate labels.
    labels_true = y_true.argmax(axis=1)
    labels_pred = y_pred.argmax(axis=1)
    
    # Calculate accuracy.
    n_good = cp.sum(labels_true - labels_pred == 0)
    return n_good / len(y_true)
