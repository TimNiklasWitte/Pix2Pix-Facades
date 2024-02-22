from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    lineplot_1 = sns.lineplot(data=df.loc[:, ["discriminator loss"]], ax=axes[0], legend=None)
    axes[0].set_title("Discriminator loss")
    #lineplot_1.set(ylim=(0.9, 1.4))


    lineplot_2 = sns.lineplot(data=df.loc[:, ["generator classic loss"]], ax=axes[1], legend=None)
    axes[1].set_title("Generator loss")
    #lineplot_2.set(ylim=(0.9, 1.4))


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