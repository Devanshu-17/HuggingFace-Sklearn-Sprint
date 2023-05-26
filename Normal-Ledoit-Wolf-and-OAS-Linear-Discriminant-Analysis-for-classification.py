import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS

def generate_data(n_samples, n_features):
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

def classify(n_train, n_test, n_averages, n_features_max, step):
    acc_clf1, acc_clf2, acc_clf3 = [], [], []
    n_features_range = range(1, n_features_max + 1, step)
    
    for n_features in n_features_range:
        score_clf1, score_clf2, score_clf3 = 0, 0, 0
        for _ in range(n_averages):
            X, y = generate_data(n_train, n_features)

            clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None).fit(X, y)
            clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
            oa = OAS(store_precision=False, assume_centered=False)
            clf3 = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa).fit(X, y)

            X, y = generate_data(n_test, n_features)
            score_clf1 += clf1.score(X, y)
            score_clf2 += clf2.score(X, y)
            score_clf3 += clf3.score(X, y)

        acc_clf1.append(score_clf1 / n_averages)
        acc_clf2.append(score_clf2 / n_averages)
        acc_clf3.append(score_clf3 / n_averages)

    features_samples_ratio = np.array(n_features_range) / n_train

    plt.plot(
        features_samples_ratio,
        acc_clf1,
        linewidth=2,
        label="LDA",
        color="gold",
        linestyle="solid",
    )
    plt.plot(
        features_samples_ratio,
        acc_clf2,
        linewidth=2,
        label="LDA with Ledoit Wolf",
        color="navy",
        linestyle="dashed",
    )
    plt.plot(
        features_samples_ratio,
        acc_clf3,
        linewidth=2,
        label="LDA with OAS",
        color="red",
        linestyle="dotted",
    )

    plt.xlabel("n_features / n_samples")
    plt.ylabel("Classification accuracy")
    plt.legend(loc="lower left")
    plt.ylim((0.65, 1.0))
    plt.suptitle(
        "LDA (Linear Discriminant Analysis) vs. "
        + "\n"
        + "LDA with Ledoit Wolf vs. "
        + "\n"
        + "LDA with OAS (1 discriminative feature)"
    )
    
    # Convert the plot to Gradio compatible format
    plt.tight_layout()
    plt.savefig("plot.png")
    return "plot.png"

# Define the input and output interfaces
inputs = [
    gr.inputs.Slider(minimum=1, maximum=100, step=1, label="n_train", default=20),
    gr.inputs.Slider(minimum=1, maximum=500, step=1, label="n_test", default=200),
    gr.inputs.Slider(minimum=1, maximum=100, step=1, label="n_averages", default=50),
    gr.inputs.Slider(minimum=1, maximum=100, step=1, label="n_features_max", default=75),
    gr.inputs.Slider(minimum=1, maximum=20, step=1, label="step", default=4),
]
output = gr.outputs.Image(type="pil")
examples = [
    [20, 200, 50, 75, 4],
    [30, 250, 60, 80, 5],
    [40, 300, 70, 90, 6],
]

# Create the Gradio app
title = "Normal, Ledoit-Wolf and OAS Linear Discriminant Analysis for classification"
description = "This example illustrates how the Ledoit-Wolf and Oracle Shrinkage Approximating (OAS) estimators of covariance can improve classification. See the original example: https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html"
gr.Interface(classify, inputs, output, examples=examples, title=title, description=description).launch()
