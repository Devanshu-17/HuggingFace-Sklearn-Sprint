import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct


def classify_xor_dataset(kernel_name):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(200, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # fit the model
    fig, ax = plt.subplots(figsize=(10, 5))
    kernels = [1.0 * RBF(length_scale=1.15), 1.0 * DotProduct(sigma_0=1.0) ** 2]
    kernel_idx = 0 if kernel_name == "RBF" else 1
    kernel = kernels[kernel_idx]
    clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X, Y)

    # plot the decision function for each datapoint on the grid
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)

    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=["k"])
    ax.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
    ax.set_xticks(())
    ax.set_yticks(())
    ax.axis([-3, 3, -3, 3])
    ax.set_title(
        "%s\n Log-Marginal-Likelihood:%.3f"
        % (clf.kernel_, clf.log_marginal_likelihood(clf.kernel_.theta)),
        fontsize=12,
    )

    fig.canvas.draw()
    pil_image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return pil_image


title = "Gaussian Process Classification on the XOR Dataset"
description = "This example illustrates GPC on XOR data. Compared are a stationary, isotropic kernel (RBF) and a non-stationary kernel (DotProduct). On this particular dataset, the DotProduct kernel obtains considerably better results because the class-boundaries are linear and coincide with the coordinate axes. In general, stationary kernels often obtain better results. See the original scikit-learn example at https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_xor.html"
kernel_options = ["RBF", "DotProduct"]
iface = gr.Interface(
    classify_xor_dataset,
    gr.inputs.Radio(choices=kernel_options, label="Kernel"),
    gr.outputs.Image(label="Decision Boundary", type="pil"),
    title=title,
    description=description,
    theme="default",
    layout="vertical",
    analytics_enabled=False
)

iface.launch()
