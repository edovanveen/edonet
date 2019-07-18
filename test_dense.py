import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import edonet


# Make test dataset.
def make_dataset():
    x, y = make_moons(n_samples=1200, noise=0.2, random_state=0)
    x[:, 0] = (x[:, 0] - 0.5)/2
    x[:, 1] = (x[:, 1] - 0.25)/1.5
    encoder = [[1, 0], [0, 1]]
    y = np.array([encoder[i] for i in y])
    return train_test_split(x, y, random_state=0)


# Check accuracy.
def accuracy(y_true, y_pred):
    labels_true = y_true.argmax(axis=1)
    labels_pred = y_pred.argmax(axis=1)
    n_good = np.sum(labels_true - labels_pred == 0)
    return n_good / len(y_true)
    
    
# Make grid.
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Plot decision boundary.
def plot_contours(ax, model, xx, yy, **params):
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).argmax(axis=1)
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z, **params)
    return out
    
    
# Show decision boundary and scatter dataset.
def show_data_and_decision(model, x, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    x0, x1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(x0, x1)
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x0, x1, c=y.argmax(axis=1), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()


# Make and test model.
def main():
    
    # Make dataset.
    x_train, x_test, y_train, y_test = make_dataset()
    
    # Make and train model.
    model = edonet.NeuralNet(input_size=2,
                             layers=({'type': 'dense', 'nr_nodes': 16, 'activation': 'relu'},
                                     {'type': 'dense', 'nr_nodes': 16, 'activation': 'tanh'},
                                     {'type': 'dense', 'nr_nodes': 16, 'activation': 'relu'},
                                     {'type': 'dense', 'nr_nodes': 2, 'activation': 'softmax'}),
                             loss='CEL',
                             seed=0)
    model.fit(x_train, y_train, epochs=100, learning_rate=0.1, batch_size=10)

    # Show result on test set.
    print("accuracy: ", accuracy(y_test, model.predict(x_test)))
    show_data_and_decision(model, x_test, y_test)    
    

if __name__ == "__main__":
    main()
