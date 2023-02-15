import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

colors = sns.color_palette("Paired", n_colors=12).as_hex()


from GM import GaussianMixture


def main():
    n_components = 1
    n, d = 100, 2

    data = []
    for i in range(n_components):
        mean_x = torch.normal(mean=torch.tensor(5.0), std=torch.tensor(1.0))
        mean_y = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(3.0))
        std_x = torch.normal(mean=torch.tensor(1.0)).abs()
        std_y = torch.normal(mean=torch.tensor(1.0)).abs()

        x_ = torch.normal(mean_x, std_x, size=(n, 1))
        y_ = torch.normal(mean_y, std_y, size=(n, 1))

        data.append(torch.cat([x_, y_], 1))

    data = torch.cat(data, 0)
    print(data.shape)
    # Next, the Gaussian mixture is instantiated and ..

    model = GaussianMixture(n_components, d)
    model.fit_em(data)

    # .. used to predict the data points as they where shifted
    y = model.predict(data)
    # model.set_marginal(indices=[0])
    x1 = model.predict(data[:, 0], marginals=[0])
    # model.set_marginal(indices=[1])
    x2 = model.predict(data[:, 1], marginals=[1])

    plot(data, y, x1, x2, n)

    # model.set_marginal(indices=[])
    data, y = model.sample(n * n_components)
    x1, _ = model.sample(n * n_components, marginals=[0])
    x2, _ = model.sample(n * n_components, marginals=[1])
    plot(data, y, x1, x2, n, sample=True)


def plot(data, y, x1, x2, n, sample=False):

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    markers = ["1", "p", "*", "+", "d", "s", "h"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].set_facecolor("#bbbbbb")
    ax[0].set_xlabel("Dimension X")
    ax[0].set_ylabel("Dimension Y")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        index = i // n
        ax[0].scatter(*point, color="k", s=9, alpha=0.5, marker=markers[index])
        ax[0].scatter(
            *point, color="white", s=2, alpha=0.5, edgecolors=colors[int(y[i])]
        )

    anchored_text = AnchoredText("Prediction - color \nCluster - shape ", loc=2)
    ax[0].add_artist(anchored_text)

    ax[1].hist(x1.detach().numpy())
    ax[1].set_title(f"marginal X {'class predictions' if not sample else ''}")

    ax[2].hist(x2.detach().numpy())
    ax[2].set_title(f"marginal Y {'class predictions' if not sample else ''}")

    plt.tight_layout()
    plt.savefig(f"example_em_{'sample' if sample else ''}.pdf")


if __name__ == "__main__":
    main()
