from LoadDataframe import *
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

import seaborn as sns
import numpy as np


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    df1 = df.rename(columns={"discriminator fake accuracy": "not smoothed"})
    lineplot_1 = sns.lineplot(data=df1.loc[:, ["not smoothed"]], ax=axes[0])
    axes[0].set_title("Discriminator fake accuracy")
    data = df.loc[:, ["discriminator fake accuracy"]].to_numpy()
    data = np.reshape(data, newshape=(-1))
   
    smoothed = savgol_filter(data, 51, 3)
    axes[0].plot(smoothed, label="smoothed")
    axes[0].legend()
    axes[0].set_ylabel("Accuracy")
    lineplot_1.set(ylim=(0.75, 1))

    df1 = df.rename(columns={"discriminator real accuracy": "not smoothed"})
    lineplot_2 = sns.lineplot(data=df1.loc[:, ["not smoothed"]], ax=axes[1])
    axes[1].set_title("Discriminator real accuracy")
    data = df.loc[:, ["discriminator real accuracy"]].to_numpy()
    data = np.reshape(data, newshape=(-1))
   
    smoothed = savgol_filter(data, 51, 3)
    axes[1].plot(smoothed, label="smoothed")
    axes[1].legend()
    axes[1].set_ylabel("Accuracy")
    lineplot_2.set(ylim=(0.75, 1))

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/DiscriminatorAccuracy.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")