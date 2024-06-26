try:
    CUPY = True
    import cupy as cp
except ImportError:
    CUPY = False
    import numpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import edonet


# Make test dataset.
def make_dataset():
    x, y = make_moons(n_samples=4000, noise=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    encoder = [[1, 0], [0, 1]]
    y = np.array([encoder[i] for i in y])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    x_train = cp.array(x_train, dtype=cp.float32)
    x_test = cp.array(x_test, dtype=cp.float32)
    y_train = cp.array(y_train, dtype=cp.float32)
    y_test = cp.array(y_test, dtype=cp.float32)
    return x_train, x_test, y_train, y_test
    

# Make grid.
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Plot decision boundary.
def plot_contours(ax, model, xx, yy, **params):
    z = model.predict(cp.c_[xx.ravel(), yy.ravel()]).argmax(axis=1)
    if CUPY:
        z = cp.asnumpy(z)
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z, **params)
    return out
    
    
# Show decision boundary and scatter dataset.
def show_data_and_decision(model, x, y):
    _, ax = plt.subplots(figsize=(8, 6))
    x0, x1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(x0, x1)
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x0, x1, c=y.argmax(axis=1), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()


# Make and test model.
def test_moons_dataset():
    
    # Make dataset.
    x_train, x_test, y_train, y_test = make_dataset()
    
    # Make and train model.
    model = edonet.NeuralNet(input_size=2,
                             layers=({'type': 'Dense', 'nr_nodes': 8, 'activation': 'relu'},
                                     {'type': 'Dense', 'nr_nodes': 8, 'activation': 'tanh'},
                                     {'type': 'Dense', 'nr_nodes': 8, 'activation': 'sigmoid'},
                                     {'type': 'Dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)
    model.fit(x_train, y_train, epochs=10, learning_rate=0.01, batch_size=100, optimizer='Adam')

    # Show result on test set.
    accuracy = model.evaluate(x_test, y_test)
    assert(accuracy > 0.95)
