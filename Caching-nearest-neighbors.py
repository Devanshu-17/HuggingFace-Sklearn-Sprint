import gradio as gr
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

from sklearn.neighbors import KNeighborsTransformer, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline

def classify_digits(n_neighbors):
    X, y = load_digits(return_X_y=True)
    n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    graph_model = KNeighborsTransformer(n_neighbors=max(n_neighbors_list), mode="distance")
    classifier_model = KNeighborsClassifier(metric="precomputed")

    full_model = Pipeline(
        steps=[("graph", graph_model), ("classifier", classifier_model)]
    )

    param_grid = {"classifier__n_neighbors": n_neighbors_list}
    grid_model = GridSearchCV(full_model, param_grid)
    grid_model.fit(X, y)

    # Plot the results of the grid search.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].errorbar(
        x=n_neighbors_list,
        y=grid_model.cv_results_["mean_test_score"],
        yerr=grid_model.cv_results_["std_test_score"],
    )
    axes[0].set(xlabel="n_neighbors", title="Classification accuracy")
    axes[1].errorbar(
        x=n_neighbors_list,
        y=grid_model.cv_results_["mean_fit_time"],
        yerr=grid_model.cv_results_["std_fit_time"],
        color="r",
    )
    axes[1].set(xlabel="n_neighbors", title="Fit time (with caching)")
    fig.tight_layout()

    # Save the plot to a temporary file
    with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plot_path = temp_file.name
        plt.savefig(plot_path)
    
    plt.close()

    return plot_path

# Create a Gradio interface with adjustable parameters
n_neighbors_input = gr.inputs.Slider(minimum=1, maximum=10, default=5, step=1, label="Number of Neighbors")
plot_output = gr.outputs.Image(type="pil")

iface = gr.Interface(
    fn=classify_digits,
    inputs=n_neighbors_input,
    outputs=plot_output,
    title="Digits Classifier",
    description="This example demonstrates how to precompute the k nearest neighbors before using them in KNeighborsClassifier. KNeighborsClassifier can compute the nearest neighbors internally, but precomputing them can have several benefits, such as finer parameter control, caching for multiple use, or custom implementations. See the original scikit-learn example [here](https://scikit-learn.org/stable/auto_examples/neighbors/plot_caching_nearest_neighbors.html).",
    examples=[
        ["2"],  # Example 1
        ["7"],  # Example 2
        ["4"],  # Example 3
    ]
)
iface.launch()