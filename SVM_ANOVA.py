import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def svm_anova_app(percentiles):
    X, y = load_iris(return_X_y=True)

    # Add non-informative features
    rng = np.random.RandomState(0)
    X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))

    # Create a feature-selection transform, a scaler, and an instance of SVM
    clf = Pipeline([
        ("anova", SelectPercentile(f_classif)),
        ("scaler", StandardScaler()),
        ("svc", SVC(gamma="auto")),
    ])

    score_means = []
    score_stds = []

    for p in percentiles:
        clf.set_params(anova__percentile=float(p))
        this_scores = cross_val_score(clf, X, y)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title("Performance of the SVM-Anova varying the percentile of features selected")
    plt.xticks(np.linspace(0, 100, 11, endpoint=True))
    plt.xlabel("Percentile")
    plt.ylabel("Accuracy Score")
    plt.axis("tight")

    # Save the plot to a file
    plt.savefig("plot.png")
    plt.close()

    return "plot.png"

iface = gr.Interface(
    fn=svm_anova_app,
    inputs=gr.inputs.CheckboxGroup(['1', '3', '6', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100'], label="Percentiles"),
    outputs="image",
    title="SVM-Anova Performance",
    description="This example shows how to perform univariate feature selection before running a SVC (support vector classifier) to improve the classification scores. We use the iris dataset (4 features) and add 36 non-informative features. We can find that our model achieves best performance when we select around 10 percent of features. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html"
)

iface.launch()
