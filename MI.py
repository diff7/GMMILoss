import torch
from GM import GaussianMixture


class MIGM(GaussianMixture):
    def __init__(
        self,
        n_components,
        n_features,
        covariance_type="full",
        eps=1.0e-6,
        cov_reg=1e-6,
        init_means="kmeans",
        mu_init=None,
        var_init=None,
        verbose=True,
        fit_mode="em",
        n_iter=1e2,
        delta=1e-6,
        learning_rate=1e-2,
        warm_start=False,
        device="cpu",
    ):

        super().__init__(
            n_components,
            n_features,
            covariance_type,
            eps,
            cov_reg,
            init_means,
            mu_init,
            var_init,
            verbose,
            device,
        )

        assert fit_mode in [
            "em",
            "grad",
        ], f"Unknown fit_mode: {fit_mode} is specified, possibale values are 1 - grad, 2 - em"
        self.fit_mode = fit_mode
        self.n_iter = n_iter
        self.delta = delta
        self.warm_start = warm_start
        self.learning_rate = learning_rate

    def fit(self, x):
        if self.fit_mode == "em":
            self.fit_em(x, self.delta, self.n_iter, self.warm_start)
        if self.fit_mode == "grad":
            self.fit_grad(x, self.n_iter, self.learning_rate)

    def compute_mi(self, data_joint, indices_a, indices_b):

        """
        data_joint : 2 dimenesional tensor n x k where k is a number of features, at least two
        indices_a : indices representing a first marignal subset
        indices_b : indices representing a second marignal subset

        Returns:
            mutual information : scalar
        """

        self.joint = self.logscore_samples(data_joint)
        device = data_joint.device
        sample_a = torch.index_select(data_joint, 1, torch.tensor(indices_a).to(device))
        self.a = self.logscore_samples(sample_a, indices_a)

        sample_b = torch.index_select(data_joint, 1, torch.tensor(indices_b).to(device))
        self.b = self.logscore_samples(sample_b, indices_b)

        mi = (self.joint - self.a - self.b).mean()

        self.print_verbose(f"MI COMPUTE:{mi}")

        return mi
