import gradio as gr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

colors = ["navy", "turquoise", "darkorange"]


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")


def classify_iris(cov_type):
    iris = datasets.load_iris()

    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    X_test = iris.data[test_index]
    y_test = iris.target[test_index]

    n_classes = len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    estimator = GaussianMixture(
        n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0
    )

    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array(
        [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
    )

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    fig, ax = plt.subplots(figsize=(8, 6))

    make_ellipses(estimator, ax)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        ax.scatter(data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n])

    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        ax.scatter(data[:, 0], data[:, 1], marker="x", color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    ax.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=ax.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    ax.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=ax.transAxes)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(cov_type.capitalize())

    plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))

    # Save the plot to a file and return its path
    output_path = "classification_plot.png"
    plt.savefig(output_path)
    plt.close()

    return output_path


iface = gr.Interface(
    fn=classify_iris,
    inputs=gr.inputs.Radio(["spherical", "diag", "tied", "full"], label="Covariance Type"),
    outputs="image",
    title="Gaussian Mixture Model Covariance",
    description="Explore different covariance types for Gaussian mixture models (GMMs) in this demonstration. GMMs are commonly used for clustering, but in this example, we compare the obtained clusters with the actual classes from the dataset. By initializing the means of the Gaussians with the means of the classes in the training set, we ensure a valid comparison. The plots show the predicted labels on both training and test data using GMMs with spherical, diagonal, full, and tied covariance matrices. Interestingly, while full covariance is expected to perform best, it may overfit small datasets and struggle to generalize to held out test data. See the original scikit-learn example for more information: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html",
    examples=[
        ["spherical"],
        ["diag"],
        ["tied"],
        ["full"],
    ],
)

iface.launch()
