from LoadDataframe import *
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter

import seaborn as sns
import numpy as np

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    df1 = df.rename(columns={"discriminator loss": "not smoothed"})
    sns.lineplot(data=df1.loc[:, ["not smoothed"]], ax=axes[0])
    axes[0].set_title("Discriminator loss")
    data = df.loc[:, ["discriminator loss"]].to_numpy()
    data = np.reshape(data, newshape=(-1))
   
    smoothed = savgol_filter(data, 51, 3)
    axes[0].plot(smoothed, label="smoothed")
    axes[0].legend()
    axes[0].set_ylabel("Loss")


    df1 = df.rename(columns={"generator classic loss": "not smoothed"})
    sns.lineplot(data=df1.loc[:, ["not smoothed"]], ax=axes[1])
    axes[1].set_title("Generator loss")
    data = df.loc[:, ["generator classic loss"]].to_numpy()
    data = np.reshape(data, newshape=(-1))
   
    smoothed = savgol_filter(data, 51, 3)
    axes[1].plot(smoothed, label="smoothed")
    axes[1].legend()
    axes[1].set_ylabel("Loss")


    # grid
    for ax in axes.flatten():
        ax.grid()


    plt.tight_layout()
    plt.savefig("../plots/GeneratorDiscriminatorLoss.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")