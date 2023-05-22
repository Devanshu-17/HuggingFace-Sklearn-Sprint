import gradio as gr
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

def generate_synthetic_data(n_samples, n_features, n_redundant, n_informative, random_state, n_clusters_per_class):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=n_redundant,
        n_informative=n_informative,
        random_state=random_state,
        n_clusters_per_class=n_clusters_per_class,
    )
    return X, y

def plot_roc_det_curves(classifier_names, svm_c, rf_max_depth, rf_n_estimators, rf_max_features,
                        n_samples, n_features, n_redundant, n_informative, random_state, n_clusters_per_class):
    X, y = generate_synthetic_data(n_samples, n_features, n_redundant, n_informative, random_state, n_clusters_per_class)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    classifiers = {
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=svm_c)),
        "Random Forest": RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features
        ),
    }

    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    for classifier_name in classifier_names:
        clf = classifiers[classifier_name]
        clf.fit(X_train, y_train)
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=classifier_name)
        DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=classifier_name)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    ax_roc.grid(linestyle="--")
    ax_det.grid(linestyle="--")

    plt.legend()
    plt.tight_layout()

    return plt

parameters = [
    gr.inputs.CheckboxGroup(["Linear SVM", "Random Forest"], label="Classifiers"),
    gr.inputs.Slider(0.001, 0.1, step=0.001, default=0.025, label="Linear SVM C"),
    gr.inputs.Slider(1, 10, step=1, default=5, label="Random Forest Max Depth"),
    gr.inputs.Slider(1, 20, step=1, default=10, label="Random Forest n_estimators"),
    gr.inputs.Slider(1, 10, step=1, default=1, label="Random Forest max_features"),
    gr.inputs.Slider(100, 2000, step=100, default=1000, label="Number of Samples"),
    gr.inputs.Slider(1, 10, step=1, default=2, label="Number of Features"),
    gr.inputs.Slider(0, 10, step=1, default=0, label="Number of Redundant Features"),
    gr.inputs.Slider(1, 10, step=1, default=2, label="Number of Informative Features"),
    gr.inputs.Slider(0, 100, step=1, default=1, label="Random State"),
    gr.inputs.Slider(1, 10, step=1, default=1, label="Number of Clusters per Class"),
]

examples = [
    [
        ["Linear SVM"],
        0.025,
        5,
        10,
        1,
        1000,
        2,
        0,
        2,
        1,
        1,
    ],
     [
        ["Random Forest"],
        0.025,
        5,
        10,
        1,
        1000,
        2,
        0,
        2,
        1,
        1,
    ],
     [
        ["Linear SVM", "Random Forest"],
        0.025,
        5,
        10,
        1,
        1000,
        2,
        0,
        2,
        1,
        1,
    ]
]

iface = gr.Interface(title = "Detection error tradeoff (DET) curve", fn=plot_roc_det_curves, inputs=parameters, outputs="plot", description="In this example, we compare two binary classification multi-threshold metrics: the Receiver Operating Characteristic (ROC) and the Detection Error Tradeoff (DET). For such purpose, we evaluate two different classifiers for the same classification task. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html", examples=examples)
iface.launch()
