from math import e
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

def visualize_isotonic_regression(n, random_range_min, random_range_max, out_of_bounds):
    if random_range_min >= random_range_max:
        raise ValueError("Random Value Range (Min) must be less than Random Value Range (Max)")

    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(random_range_min, random_range_max, size=(n,)) + 50.0 * np.log1p(np.arange(n))

    ir = IsotonicRegression(out_of_bounds=out_of_bounds if out_of_bounds else "clip")
    y_ = ir.fit_transform(x, y)

    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

    segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(np.full(n, 0.5))

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

    ax0.plot(x, y, "C0.", markersize=12)
    ax0.plot(x, y_, "C1.-", markersize=12)
    ax0.plot(x, lr.predict(x[:, np.newaxis]), "C2-")
    ax0.add_collection(lc)
    ax0.legend(("Training data", "Isotonic fit", "Linear fit"), loc="lower right")
    ax0.set_title("Isotonic regression fit on noisy data (n=%d)" % n)

    x_test = np.linspace(np.min(x), np.max(x), 1000)  # Update test values range
    ax1.plot(x_test, ir.predict(x_test), "C1-")
    ax1.plot(ir.X_thresholds_, ir.y_thresholds_, "C1.", markersize=12)
    ax1.set_title("Prediction function (%d thresholds)" % len(ir.X_thresholds_))

    return fig

parameters = [
    gr.inputs.Slider(10, 100, step=10, default=50, label="Number of data points (n)"),
    gr.inputs.Slider(-50, 50, step=1, default=-50, label="Random Value Range (Min)"),
    gr.inputs.Slider(-50, 50, step=1, default=50, label="Random Value Range (Max)"),
    gr.inputs.Dropdown(["clip", "nan", "raise"], default="clip", label="Out of Bounds Strategy"),
]

description = "This app presents an illustration of the isotonic regression on generated data (non-linear monotonic trend with homoscedastic uniform noise). The isotonic regression algorithm finds a non-decreasing approximation of a function while minimizing the mean squared error on the training data. The benefit of such a non-parametric model is that it does not assume any shape for the target function besides monotonicity. For comparison a linear regression is also presented. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_isotonic_regression.html"

examples = [
    [50, -30, 30, "clip"],
    [30, -20, 40, "nan"],
    [70, -10, 20, "raise"],
]

iface = gr.Interface(fn=visualize_isotonic_regression, inputs=parameters, outputs="plot", title="Isotonic Regression Visualization", description=description, examples=examples)
iface.launch()
