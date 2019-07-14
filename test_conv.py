import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import edonet


# Make and test model.
def main():
    
    """
    # Test flatten layer.        
    np.random.seed(0)
    f = edonet.Conv2DLayer((4, 4, 3), nr_filters=10, filter_size=(2, 2), stride=(2, 2), activation='relu', padding='same')
    
    print(f.padding)
    
    x = np.array([[[1.0, 1.1, 0.0],
                   [1.2, 2.0, 3.0],
                   [4.0, 5.0, 1.3],
                   [0.1, 1.4, 1.5]],
                  [[7.0, 8.0, 1.6],
                   [0.2, 1.7, 2.1],
                   [4.1, 5.1, 1.8],
                   [0.3, 1.9, 1.11]],
                  [[1, 1, 4],
                   [1, 1, 3],
                   [4, 0, 1],
                   [0, 1, 1]],
                  [[2, 8, 1],
                   [0, 1, 2],
                   [9, 1, 1],
                   [0, 1, 1]]])
    x = np.array([x])
    
    f.init_weights()
    y = f.forward_prop(x)
    
    print(x)
    print(y)
    """
    
    # Test copied code for simple situation

    y = [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [8, 7, 6, 5, 4],
         [3, 2, 1, 0, 1],
         [2, 3, 4, 5, 6],
         [7, 8, 9, 8, 7]]
    f = [[[0 -1], 
          [1, 0]], 
         [[1, -1],
          [1, -1]]]
    y = np.array(y)
    f = np.array(f)
    filter_size = (2, 2)
    stride = (1, 1)
    output_size = (int(np.floor((y.shape[0] - filter_size[0] + 1) / stride[0])),
                   int(np.floor((y.shape[1] - filter_size[1] + 1) / stride[1])))
    print(y.shape, output_size)
    
    # Keep track of all the dimensions
    i, j = y.strides
    p, q = filter_size
    n, m, _ = output_size
    sx, sy = stride

    # Create strided submatrices.
    sub_shape = (nr_examples, p, q, n, m, c)
    sub_strides = (h, i * sx, j * sy, i * sx, j * sy, k)
    as_strided = np.lib.stride_tricks.as_strided
    sub_matrices = as_strided(y, shape=sub_shape, strides=sub_strides)

    # Get convolution.
    return np.einsum('hijklm,ijmn->hkln', sub_matrices, self.filters)
    
    
if __name__ == "__main__":
    main()
