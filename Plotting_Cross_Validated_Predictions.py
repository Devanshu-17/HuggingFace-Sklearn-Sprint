from turtle import title
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import PredictionErrorDisplay


def predict_diabetes(subsample, plot_type):
    X, y = load_diabetes(return_X_y=True)
    lr = LinearRegression()
    y_pred = cross_val_predict(lr, X, y, cv=10)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    if "Actual vs. Predicted" in plot_type:
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            subsample=subsample,
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")
    if "Residuals vs. Predicted" in plot_type:
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=y_pred,
            kind="residual_vs_predicted",
            subsample=subsample,
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")

    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.close(fig)

    # Save the figure as an image
    image_path = "predictions.png"
    fig.savefig(image_path)
    return image_path


# Define the Gradio interface
inputs = [
    gr.inputs.Slider(minimum=1, maximum=100, step=1, default=100, label="Subsample"),
    gr.inputs.CheckboxGroup(["Actual vs. Predicted", "Residuals vs. Predicted"], label="Plot Types", default=["Actual vs. Predicted", "Residuals vs. Predicted"])
]
outputs = gr.outputs.Image(label="Cross-Validated Predictions", type="pil")

title = "Plotting Cross-Validated Predictions"
description="This app plots cross-validated predictions for a linear regression model trained on the diabetes dataset. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html"
examples = [
    [
        100,
        ["Actual vs. Predicted"],
        "Plotting cross-validated predictions with Actual vs. Predicted plot.",
    ],
    [
        50,
        ["Residuals vs. Predicted"],
        "Plotting cross-validated predictions with Residuals vs. Predicted plot.",
    ],
    [
        75,
        ["Actual vs. Predicted", "Residuals vs. Predicted"],
        "Plotting cross-validated predictions with both Actual vs. Predicted and Residuals vs. Predicted plots.",
    ],
]

gr.Interface(fn=predict_diabetes, title=title, description=description, examples=examples, inputs=inputs, outputs=outputs).launch()
