import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import gradio as gr
import tempfile
import os

def generate_gaussian_mixture(n_samples):
    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0.0, -0.7], [3.5, 0.7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()
    CS = ax.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = fig.colorbar(CS, shrink=0.8, extend="both")
    ax.scatter(X_train[:, 0], X_train[:, 1], 0.8)

    ax.set_title("Negative log-likelihood predicted by a GMM")
    ax.axis("tight")
    
    # Save the plot as a temporary image file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "gmm_plot.png")
    fig.savefig(temp_file_path)
    plt.close(fig)
    
    return temp_file_path

def plot_to_image(file_path):
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    os.remove(file_path)
    return image_bytes

inputs = gr.inputs.Slider(100, 1000, step=100, default=300, label="Number of Samples")
outputs = gr.outputs.Image(type="pil", label="GMM Plot")

title = "Density Estimation for a Gaussian mixture"
description = "In this example, you can visualize the density estimation of a mixture of two Gaussians using a Gaussian Mixture Model (GMM). The data used for the model is generated from two Gaussians with distinct centers and covariance matrices. By adjusting the number of samples, you can observe how the GMM captures the underlying distribution and generates a contour plot representing the estimated density. This interactive application allows you to explore the behavior of the GMM and gain insights into the modeling of complex data distributions using mixture models. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html"
gr.Interface(generate_gaussian_mixture, inputs, outputs, title=title, description=description, postprocess=plot_to_image).launch()
