import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import gradio as gr

def plot_decision_boundary(kernel_type):
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = np.array(iris.target, dtype=int)

    h = 0.02  # step size in the mesh

    if kernel_type == "isotropic":
        kernel = 1.0 * RBF([1.0])
        clf = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    elif kernel_type == "anisotropic":
        kernel = 1.0 * RBF([1.0, 1.0])
        clf = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    else:
        return None

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))

    plt.figure(figsize=(7, 5))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" % (kernel_type.capitalize(), clf.log_marginal_likelihood(clf.kernel_.theta)))
    plt.tight_layout()
    return plt

kernel_select = gr.inputs.Radio(["isotropic", "anisotropic"], label="Kernel Type")
gr_interface = gr.Interface(fn=plot_decision_boundary, inputs=kernel_select, outputs="plot", title="Gaussian Process Classification on Iris Dataset", description="This example illustrates the predicted probability of GPC for an isotropic and anisotropic RBF kernel on a two-dimensional version for the iris-dataset. The anisotropic RBF kernel obtains slightly higher log-marginal-likelihood by assigning different length-scales to the two feature dimensions. See the original example at https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html")
gr_interface.launch()
