import gradio as gr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

EPSILON = np.finfo(np.float32).eps
n_samples = 20
seed = np.random.RandomState(seed=3)
X_true = seed.randint(0, 20, 2 * n_samples).astype(float)
X_true = X_true.reshape((n_samples, 2))
# Center the data
X_true -= X_true.mean()

similarities = euclidean_distances(X_true)

# Add noise to the similarities
noise = np.random.rand(n_samples, n_samples)
noise = noise + noise.T
noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
similarities += noise

def mds_nmds(n_components, max_iter, eps):
    mds = manifold.MDS(
        n_components=n_components,
        max_iter=max_iter,
        eps=eps,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    pos = mds.fit(similarities).embedding_

    nmds = manifold.MDS(
        n_components=n_components,
        metric=False,
        max_iter=max_iter,
        eps=eps,
        dissimilarity="precomputed",
        random_state=seed,
        n_jobs=1,
        n_init=1,
        normalized_stress="auto",
    )
    npos = nmds.fit_transform(similarities, init=pos)

    # Rescale the data
    pos *= np.sqrt((X_true**2).sum()) / np.sqrt((pos**2).sum())
    npos *= np.sqrt((X_true**2).sum()) / np.sqrt((npos**2).sum())

    # Rotate the data
    clf = PCA(n_components=2)
    X_true_transformed = clf.fit_transform(X_true)
    pos_transformed = clf.fit_transform(pos)
    npos_transformed = clf.fit_transform(npos)

    return X_true_transformed, pos_transformed, npos_transformed


def plot_similarity_scatter(similarity_threshold=50, n_components=2, max_iter=3000, eps=1e-9, cmap_name='Blues'):
    X_true_transformed, pos_transformed, npos_transformed = mds_nmds(n_components, max_iter, eps)

    fig = plt.figure()
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])

    s = 100
    plt.scatter(X_true_transformed[:, 0], X_true_transformed[:, 1], color="navy", s=s, lw=0, label="True Position")
    plt.scatter(pos_transformed[:, 0], pos_transformed[:, 1], color="turquoise", s=s, lw=0, label="MDS")
    plt.scatter(npos_transformed[:, 0], npos_transformed[:, 1], color="darkorange", s=s, lw=0, label="NMDS")
    plt.legend(scatterpoints=1, loc="best", shadow=False)

    similarities_thresholded = similarities.copy()
    similarities_thresholded[similarities_thresholded <= int(similarity_threshold)] = 0

    np.fill_diagonal(similarities_thresholded, 0)
    # Plot the edges
    start_idx, end_idx = np.where(pos_transformed)
    segments = [[X_true_transformed[i, :], X_true_transformed[j, :]] for i in range(len(pos_transformed)) for j in range(len(pos_transformed))]
    values = np.abs(similarities_thresholded)
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.get_cmap(cmap_name), norm=plt.Normalize(0, values.max()))
    lc.set_array(similarities_thresholded.flatten())
    lc.set_linewidths(np.full(len(segments), 0.5))
    ax.add_collection(lc)

    # Save the plot as a PNG file
    plt.savefig("plot.png")
    plt.close()

    # Return the saved plot file
    return "plot.png"



parameters = [
    gr.inputs.Slider(label="Similarity Threshold", minimum=0, maximum=100, step=1, default=50),
    gr.inputs.Slider(label="Number of Components", minimum=1, maximum=10, step=1, default=2),
    gr.inputs.Slider(label="Max Iterations", minimum=100, maximum=5000, step=100, default=3000),
    gr.inputs.Slider(label="Epsilon", minimum=1e-12, maximum=1e-6, step=1e-12, default=1e-9),
    gr.inputs.Dropdown(label="Colormap", choices=["Blues_r", "Dark2", "Reds_r", "Purples_r"], default="Blues_r")
]


iface = gr.Interface(
    fn=plot_similarity_scatter,
    inputs=parameters,
    outputs="image",
    title="Multi-dimensional scaling",
    description="The scatter plot is generated based on the provided data and similarity matrix. MDS and NMDS techniques are used to project the data points into a two-dimensional space. The points are plotted in the scatter plot, with different colors representing the true positions, MDS positions, and NMDS positions of the data points. The similarity threshold parameter allows you to control the visibility of connections between the points. Points with similarity values below the threshold are not connected by lines in the plot.  See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html",
    examples=[
    [50, 2, 3000, 1e-9, "Blues_r"],
    [75, 3, 2000, 1e-10, "Dark2"],
    [90, 2, 4000, 1e-11, "Reds_r"],
],

)

iface.launch()
