import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

def generate_digits(bandwidth, num_samples):

    # convert bandwidth to integer
    bandwidth = int(bandwidth)

    # convert num_samples to integer
    num_samples = int(num_samples)

    # load the data
    digits = load_digits()

    # project the 64-dimensional data to a lower dimension
    pca = PCA(n_components=15, whiten=False)
    data = pca.fit_transform(digits.data)

    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)

    # use the specified bandwidth to compute the kernel density estimate
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(data)

    # sample new points from the data
    new_data = kde.sample(num_samples, random_state=0)
    new_data = pca.inverse_transform(new_data)

    # reshape the data into a 4x11 grid
    new_data = new_data.reshape((num_samples, 64))
    real_data = digits.data[:num_samples].reshape((num_samples, 64))

    # create the plot
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            index = i * 11 + j  # Calculate the correct index
            if index < num_samples:
                im = ax[i, j].imshow(
                    real_data[index].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
                )
                im.set_clim(0, 16)
                im = ax[i + 5, j].imshow(
                    new_data[index].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
                )
                im.set_clim(0, 16)
            else:
                ax[i, j].axis("off")
                ax[i + 5, j].axis("off")

    ax[0, 5].set_title("Selection from the input data")
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')


    # save the plot to a file
    plt.savefig("digits_plot.png")

    # return the path to the generated plot
    return "digits_plot.png"

# create the Gradio interface
inputs = [
    gr.inputs.Slider(minimum=1, maximum=10, step=1, label="Bandwidth"),
    gr.inputs.Number(default=44, label="Number of Samples")
]
output = gr.outputs.Image(type="pil")

title = "Kernel Density Estimation"
description = "This example shows how kernel density estimation (KDE), a powerful non-parametric density estimation technique, can be used to learn a generative model for a dataset. With this generative model in place, new samples can be drawn. These new samples reflect the underlying model of the data. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html"
examples = [
    [1, 44],  # Changed to integer values
    [8, 22],  # Changed to integer values
    [7, 51]  # Changed to integer values
]

gr.Interface(generate_digits, inputs, output, title=title, description=description, examples=examples).launch()
