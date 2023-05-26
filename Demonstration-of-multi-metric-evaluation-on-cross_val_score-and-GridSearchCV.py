import gradio as gr
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import tempfile


def grid_search(min_samples_split, max_depth):
    X, y = make_hastie_10_2(n_samples=8000, random_state=42)
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={"min_samples_split": range(min_samples_split, max_depth + 1, 20)},
        scoring=scoring,
        refit="AUC",
        n_jobs=2,
        return_train_score=True,
    )
    gs.fit(X, y)
    results = gs.cv_results_

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    ax = plt.gca()
    ax.set_xlim(min_samples_split, max_depth)
    ax.set_ylim(0.73, 1)

    X_axis = np.array(results["param_min_samples_split"].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)

    # Save the plot as an image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_filename = temp_file.name
        plt.savefig(temp_filename)

    # Return the path to the image file
    return temp_filename


min_samples_split_input = gr.inputs.Slider(minimum=2, maximum=402, default=2, step=20, label="min_samples_split")
max_depth_input = gr.inputs.Slider(minimum=2, maximum=402, default=402, step=20, label="max_depth")
outputs = gr.outputs.Image(type="pil", label="Score Plot")

title = "Multi-Metric Evaluation on Cross_Val_Score and GridSearchCV"
description = "This app allows users to explore the performance of a Decision Tree Classifier by adjusting the parameters 'min_samples_split' and 'max_depth'. The app performs a grid search and evaluates the classifier using multiple scoring metrics. The resulting score plot provides insights into the impact of parameter variations on model performance. Users can interactively modify the parameter values using sliders and observe the corresponding changes in the score plot. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html"
examples = [
    [42, 402],
    [130, 340],
    [88, 240],
]

gr.Interface(fn=grid_search, inputs=[min_samples_split_input, max_depth_input], outputs=outputs,
              title=title, description=description, examples=examples).launch()
