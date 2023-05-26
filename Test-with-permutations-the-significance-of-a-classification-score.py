import gradio as gr
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, permutation_test_score
import numpy as np
import tempfile
import os

def run_permutation_test(display_option, kernel, random_state, n_permutations):
    iris = load_iris()
    X = iris.data
    y = iris.target

    n_uncorrelated_features = 20
    rng = np.random.RandomState(seed=0)
    X_rand = rng.normal(size=(X.shape[0], n_uncorrelated_features))

    clf = SVC(kernel=kernel, random_state=random_state)
    cv = StratifiedKFold(2, shuffle=True, random_state=0)

    score_iris, perm_scores_iris, pvalue_iris = permutation_test_score(
        clf, X, y, scoring="accuracy", cv=cv, n_permutations=n_permutations
    )

    score_rand, perm_scores_rand, pvalue_rand = permutation_test_score(
        clf, X_rand, y, scoring="accuracy", cv=cv, n_permutations=n_permutations
    )

    original_plot_path = None
    random_plot_path = None

    if display_option in ['original', 'both']:
        # Original data
        fig, ax = plt.subplots()
        ax.hist(perm_scores_iris, bins=20, density=True)
        ax.axvline(score_iris, ls="--", color="r")
        score_label = f"Score on original\ndata: {score_iris:.2f}\n(p-value: {pvalue_iris:.3f})"
        ax.text(0.7, 10, score_label, fontsize=12)
        ax.set_xlabel("Accuracy score")
        ax.set_ylabel("Probability")
        original_plot_path = os.path.join(tempfile.mkdtemp(), "original_plot.png")
        plt.savefig(original_plot_path)
        plt.close()

    if display_option in ['random', 'both']:
        # Random data
        fig, ax = plt.subplots()
        ax.hist(perm_scores_rand, bins=20, density=True)
        ax.set_xlim(0.13)
        ax.axvline(score_rand, ls="--", color="r")
        score_label = f"Score on original\ndata: {score_rand:.2f}\n(p-value: {pvalue_rand:.3f})"
        ax.text(0.14, 7.5, score_label, fontsize=12)
        ax.set_xlabel("Accuracy score")
        ax.set_ylabel("Probability")
        random_plot_path = os.path.join(tempfile.mkdtemp(), "random_plot.png")
        plt.savefig(random_plot_path)
        plt.close()

    return original_plot_path, random_plot_path

iface = gr.Interface(
    fn=run_permutation_test,
    inputs=[
        gr.inputs.Dropdown(
            choices=["original", "random", "both"],
            label="Display Option",
            default="both"
        ),
        gr.inputs.Dropdown(
            choices=["linear", "rbf", "poly"],
            label="Kernel",
            default="linear"
        ),
        gr.inputs.Slider(
            minimum=0, maximum=10, step=1,
            label="Random State",
            default=7
        ),
        gr.inputs.Slider(
            minimum=100, maximum=2000, step=100,
            label="Number of Permutations",
            default=1000
        )
    ],
    outputs=["image", "image"],
    title="Test with permutations the significance of a classification score",
    description="This example demonstrates the use of permutation_test_score to evaluate the significance of a cross-validated score using permutations. This operation is being performed on the Iris Dataset. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html",
    examples=[
        ["both", "linear", 7, 1000],
        ["original", "rbf", 3, 500],
        ["random", "poly", 5, 1500]
    ],
    allow_flagging=False
)
iface.launch()
