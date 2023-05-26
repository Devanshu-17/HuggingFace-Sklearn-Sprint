import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import row_norms
from sklearn.datasets._samples_generator import make_blobs
from timeit import default_timer as timer

def initialize_gmm(X, method, n_components, max_iter):
    n_samples = X.shape[0]
    x_squared_norms = row_norms(X, squared=True)

    def get_initial_means(X, init_params, r):
        gmm = GaussianMixture(
            n_components=n_components, init_params=init_params, tol=1e-9, max_iter=0, random_state=r
        ).fit(X)
        return gmm.means_

    r = np.random.RandomState(seed=1234)
    start = timer()
    ini = get_initial_means(X, method, r)
    end = timer()
    init_time = end - start

    gmm = GaussianMixture(
        n_components=n_components, means_init=ini, tol=1e-9, max_iter=max_iter, random_state=r
    ).fit(X)

    return gmm, ini, init_time


def visualize_gmm(X, gmm, ini, init_time, n_components):
    methods = ["kmeans", "random_from_data", "k-means++", "random"]
    colors = ["navy", "turquoise", "cornflowerblue", "darkorange"]
    times_init = {}
    relative_times = {}

    plt.figure(figsize=(4 * len(methods) // 2, 6))
    plt.subplots_adjust(
        bottom=0.1, top=0.9, hspace=0.15, wspace=0.05, left=0.05, right=0.95
    )

    for n, method in enumerate(methods):
        r = np.random.RandomState(seed=1234)
        plt.subplot(2, len(methods) // 2, n + 1)

        gmm = GaussianMixture(
            n_components=n_components, means_init=ini, tol=1e-9, max_iter=gmm.n_iter_, random_state=r
        ).fit(X)

        times_init[method] = init_time
        for i, color in enumerate(colors):
            data = X[gmm.predict(X) == i]
            plt.scatter(data[:, 0], data[:, 1], color=color, marker="x")

        plt.scatter(
            ini[:, 0], ini[:, 1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black"
        )
        relative_times[method] = times_init[method] / times_init[methods[0]]

        plt.xticks(())
        plt.yticks(())
        plt.title(method, loc="left", fontsize=12)
        plt.title(
            "Iter %i | Init Time %.2fx" % (gmm.n_iter_, relative_times[method]),
            loc="right",
            fontsize=10,
        )

    return plt

# Generate some data
X, y_true = make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]

def run_gmm(method, n_components=4, max_iter=2000):
    gmm, ini, init_time = initialize_gmm(X, method, int(n_components), int(max_iter))
    plot = visualize_gmm(X, gmm, ini, init_time, int(n_components))
    return plot

iface = gr.Interface(
    fn=run_gmm,
    title="Gaussian Mixture Model Initialization Methods",
    description="GMM Initialization Methods is a visualization tool showcasing different initialization methods in Gaussian Mixture Models. The example demonstrates four initialization approaches: kmeans (default), random, random_from_data, and k-means++. The plot displays orange diamonds representing the initialization centers for each method, while crosses represent the data points with color-coded classifications after GMM convergence. The numbers in the subplots indicate the iteration count and relative initialization time. Alternative methods show lower initialization times but may require more iterations to converge. Notably, k-means++ achieves a good balance of fast initialization and convergence. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_init.html",
    inputs=[
        gr.inputs.Dropdown(["kmeans", "random_from_data", "k-means++", "random"], label="Method", default="kmeans"),
        gr.inputs.Number(default=4, label="Number of Components"),
        gr.inputs.Number(default=2000, label="Max Iterations")
    ],
    outputs="plot",
    examples=[
        ["kmeans", 4, 2000],
        ["random_from_data", 3, 1000],
        ["k-means++", 8, 1000],
        ["random", 11, 1000],
    ],
)

iface.launch()
