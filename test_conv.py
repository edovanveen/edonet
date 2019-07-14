import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import edonet


# Make and test model.
def main():
    
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
    

if __name__ == "__main__":
    main()
