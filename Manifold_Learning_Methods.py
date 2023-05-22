import gradio as gr
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D


def compare_manifold_learning(methods, n_samples, n_neighbors, n_components, perplexity):
    S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
    transformed_data = []

    if len(methods) == 1:
        method = methods[0]
        manifold_method = {
            "Locally Linear Embeddings Standard": manifold.LocallyLinearEmbedding(method="standard", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
            "Locally Linear Embeddings LTSA": manifold.LocallyLinearEmbedding(method="ltsa", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
            "Locally Linear Embeddings Hessian": manifold.LocallyLinearEmbedding(method="hessian", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
            "Locally Linear Embeddings Modified": manifold.LocallyLinearEmbedding(method="modified", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
            "Isomap": manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1),
            "MultiDimensional Scaling": manifold.MDS(n_components=n_components, max_iter=50, n_init=4, random_state=0, normalized_stress=False),
            "Spectral Embedding": manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors),
            "T-distributed Stochastic Neighbor Embedding": manifold.TSNE(n_components=n_components, perplexity=perplexity, init="random", n_iter=250, random_state=0)
        }[method]
        S_transformed = manifold_method.fit_transform(S_points)
        transformed_data.append(S_transformed)
    else:
        for method in methods:
            manifold_method = {
                "Locally Linear Embeddings Standard": manifold.LocallyLinearEmbedding(method="standard", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
                "Locally Linear Embeddings LTSA": manifold.LocallyLinearEmbedding(method="ltsa", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
                "Locally Linear Embeddings Hessian": manifold.LocallyLinearEmbedding(method="hessian", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
                "Locally Linear Embeddings Modified": manifold.LocallyLinearEmbedding(method="modified", n_neighbors=n_neighbors, n_components=n_components, eigen_solver="auto", random_state=0),
                "Isomap": manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1),
                "MultiDimensional Scaling": manifold.MDS(n_components=n_components, max_iter=50, n_init=4, random_state=0, normalized_stress=False),
                "Spectral Embedding": manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors),
                "T-distributed Stochastic Neighbor Embedding": manifold.TSNE(n_components=n_components, perplexity=perplexity, init="random", n_iter=250, random_state=0)
            }[method]
            S_transformed = manifold_method.fit_transform(S_points)
            transformed_data.append(S_transformed)

    fig, axs = plt.subplots(1, len(transformed_data), figsize=(6 * len(transformed_data), 6))
    fig.suptitle("Manifold Learning Comparison", fontsize=16)

    if len(methods) == 1:
        ax = axs
        method = methods[0]
        data = transformed_data[0]
        ax.scatter(data[:, 0], data[:, 1], c=S_color, cmap=plt.cm.Spectral)
        ax.set_title(f"Method: {method}")
        ax.axis("tight")
        ax.axis("off")
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        for ax, method, data in zip(axs, methods, transformed_data):
            ax.scatter(data[:, 0], data[:, 1], c=S_color, cmap=plt.cm.Spectral)
            ax.set_title(f"Method: {method}")
            ax.axis("tight")
            ax.axis("off")
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

    return "plot.png"

method_options = [
    "Locally Linear Embeddings Standard",
    "Locally Linear Embeddings LTSA",
    "Locally Linear Embeddings Hessian",
    "Locally Linear Embeddings Modified",
    "Isomap",
    "MultiDimensional Scaling",
    "Spectral Embedding",
    "T-distributed Stochastic Neighbor Embedding"
]

inputs = [
    gr.components.CheckboxGroup(method_options, label="Manifold Learning Methods"),
    gr.inputs.Slider(default=1500, label="Number of Samples", maximum=5000),
    gr.inputs.Slider(default=12, label="Number of Neighbors"),
    gr.inputs.Slider(default=2, label="Number of Components"),
    gr.inputs.Slider(default=30, label="Perplexity (for t-SNE)")
]

gr.Interface(
    fn=compare_manifold_learning,
    inputs=inputs,
    outputs="image",
    examples=[
        [method_options, 1500, 12, 2, 30]
    ],
    title="Manifold Learning Comparison",
    description="This code demonstrates a comparison of manifold learning methods using the S-curve dataset. Manifold learning techniques aim to uncover the underlying structure and relationships within high-dimensional data by projecting it onto a lower-dimensional space. This comparison allows you to explore the effects of different methods on the dataset. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html"
).launch()
