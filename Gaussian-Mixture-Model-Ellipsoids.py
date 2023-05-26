import itertools
import gradio as gr
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def generate_plot(num_components_gmm, num_components_dpgmm):
    num_components_gmm = int(num_components_gmm)
    num_components_dpgmm = int(num_components_dpgmm)
    np.random.seed(0)
    C = np.array([[0.0, -0.1], [1.7, 0.4]])
    X = np.r_[
        np.dot(np.random.randn(n_samples, 2), C),
        0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
    ]
    gmm = mixture.GaussianMixture(n_components=num_components_gmm, covariance_type="full").fit(X)
    dpgmm = mixture.BayesianGaussianMixture(n_components=num_components_dpgmm, covariance_type="full").fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")
    plot_results(
        X,
        dpgmm.predict(X),
        dpgmm.means_,
        dpgmm.covariances_,
        1,
        "Bayesian Gaussian Mixture with a Dirichlet process prior",
    )
    plt.tight_layout()
    # Save the plot as an image file
    image_path = "plot.png"
    plt.savefig(image_path)
    plt.close()  # Close the plot to release memory
    return image_path

n_samples = 500

iface = gr.Interface(
    generate_plot,
    [
        gr.inputs.Slider(1, 10, 1, label="Number of components (GMM)"),
        gr.inputs.Slider(1, 10, 1, label="Number of components (DPGMM)"),
    ],
    gr.outputs.Image(type="pil"),
    title="Gaussian Mixture Model Ellipsoids",
    description="Gaussian Mixture Model Ellipsoids is an example that demonstrates the use of Expectation Maximization (GaussianMixture class) and Variational Inference (BayesianGaussianMixture class) models to fit a mixture of two Gaussians. The models have access to five components for fitting the data, but the Expectation Maximization model uses all five components while the Variational Inference model adapts the number of components based on the data. The plot shows that the Expectation Maximization model may split components arbitrarily when trying to fit too many components. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html",
    examples=[
        ["5", "5"],
        ["3", "7"],
        ["2", "4"],
    ],
)

iface.launch()
