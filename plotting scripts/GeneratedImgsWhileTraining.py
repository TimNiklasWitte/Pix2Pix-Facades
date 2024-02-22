from LoadDataframe import *
from matplotlib import pyplot as plt

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    img_list = []
    for epoch in range(0, 300 + 10, 10):
        imgs = df.loc[epoch, "generated imgs"]
        img = imgs[0]

        fig, axes = plt.subplots(nrows=1, ncols=3)

        input_img = img[0:, :256, :]
        generated_img = img[0:, 256:512, :]
        ground_truth = img[0:, 512:, :]
        
        axes[0].imshow(input_img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(generated_img)
        axes[1].set_title("Generated image")
        axes[1].axis("off")

        axes[2].imshow(ground_truth)
        axes[2].set_title("Ground truth")
        axes[2].axis("off")

        fig.suptitle(f"Epoch: {epoch}")
        plt.tight_layout()
        plt.savefig(f"../plots/generated images while training/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")