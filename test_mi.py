import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

matplotlib.use("Agg")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

from MI import MIGM
from sklearn.feature_selection import mutual_info_regression


def main():

    n = 50
    r = np.linspace(0.1, 0.99, 30)

    ksg_list = []
    mia_list = []
    migm_list = []

    for ri in r:
        mu = torch.tensor([0.0, 0.0]).float()
        rit = torch.tensor(ri).float()
        var = torch.tensor([[1.0, rit], [rit, 1.0]])
        sampler = torch.distributions.multivariate_normal.MultivariateNormal(mu, var)

        data = torch.stack([sampler.sample() for _ in range(n)])
        model = MIGM(1, 2, init_means="kmeans")
        model.fit(data)
        mi = model.compute_mi(data, [0], [1])
        migm_list.append(mi.detach().numpy())

        mi_a = -0.5 * torch.log(torch.tensor(1 - ri**2))
        print(f"MI analytical from estinated var: {mi_a}, p-corr {ri}")
        mia_list.append(mi_a)

        mi_ksg = mutual_info_regression(
            data[:, 1].numpy().reshape(-1, 1), data[:, 0].numpy()
        )
        print(f"MI KSG: {mi_ksg}")
        ksg_list.append(mi_ksg)

    plot(r, ksg_list, mia_list, migm_list, n)


def plot(r, ksg_list, mia_list, migm_list, samples):

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    if len(mia_list) > 0:
        ax.plot(r, mia_list, label="Analytical", c="k", lw=3, ls="-.")

    ax.plot(r, ksg_list, label="KSG", c="tomato", lw=3)
    ax.plot(r, migm_list, label="GM", c="g", lw=3)

    ax.set_xlabel(r"$\rho$", fontsize=30)
    ax.set_ylabel(f"MI [nat] | Set size: {samples}", fontsize=25)
    plt.legend()
    plt.savefig(f"MI_simple_normal_for_#_{samples}.pdf")


if __name__ == "__main__":
    main()
