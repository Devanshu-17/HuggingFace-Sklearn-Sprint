from turtle import title
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_species_distributions
from sklearn.neighbors import KernelDensity
import gradio as gr

def construct_grids(batch):
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    ygrid = np.arange(ymin, ymax, batch.grid_size)
    return (xgrid, ygrid)

def plot_species_distributions(bandwidth):
    data = fetch_species_distributions()
    species_names = ["Bradypus Variegatus", "Microryzomys Minutus"]
    Xtrain = np.vstack([data["train"]["dd lat"], data["train"]["dd long"]]).T
    ytrain = np.array(
        [d.decode("ascii").startswith("micro") for d in data["train"]["species"]],
        dtype="int",
    )
    Xtrain *= np.pi / 180.0

    xgrid, ygrid = construct_grids(data)
    X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
    land_reference = data.coverages[6][::5, ::5]
    land_mask = (land_reference > -9999).ravel()

    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = xy[land_mask]
    xy *= np.pi / 180.0

    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        print(" - computing KDE in spherical coordinates")
        kde = KernelDensity(
            bandwidth=bandwidth, metric="haversine", kernel="gaussian", algorithm="ball_tree"
        )
        kde.fit(Xtrain[ytrain == i])
        Z = np.full(land_mask.shape[0], -9999, dtype="int")
        Z[land_mask] = np.exp(kde.score_samples(xy))
        Z = Z.reshape(X.shape)
        levels = np.linspace(0, Z.max(), 25)
        plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
        plt.contour(
            X, Y, land_reference, levels=[-9998], colors="k", linestyles="solid"
        )
        plt.xticks([])
        plt.yticks([])
        plt.title(species_names[i])

    return plt

bandwidth_input = gr.inputs.Slider(minimum=0.01, maximum=0.3, default=0.01, step=0.01, label="Bandwidth")
title="Kernel Density Estimate of Species Distributions"
description="This shows an example of a neighbors-based query (in particular a kernel density estimate) on geospatial data, using a Ball Tree built upon the Haversine distance metric – i.e. distances over points in latitude/longitude. The dataset is provided by Phillips et. al. (2006). If available, the example uses basemap to plot the coast lines and national boundaries of South America. See the original scikit-learn example here: https://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html"
iface = gr.Interface(fn=plot_species_distributions, title = title, description=description, inputs=bandwidth_input, outputs="plot")
iface.launch()
