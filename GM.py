import torch
import numpy as np
from math import pi
from copy import copy


class MarginalContext:
    def __init__(self, gm_object, indices=[]):
        self.gm_object = gm_object
        self.indices = indices

    def __enter__(self):
        if len(self.indices) > 0:
            self.gm_object._set_marginal(self.indices)

    def __exit__(self, *exc):
        self.gm_object._set_full()


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        if mat_b.shape[2] == 1:
            res[:, i, :, :] = (mat_a_i * mat_b_i).unsqueeze(1)
        else:
            res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(
        self,
        n_components,
        n_features,
        covariance_type="full",
        eps=1.0e-6,
        init_means="random",
        mu_init=None,
        var_init=None,
        device="cpu",
        verbose=True,
    ):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_means = init_means

        assert self.covariance_type in ["full", "diag"]
        assert self.init_means in ["kmeans", "random"]

        self.device = device
        self.verbose = verbose
        self._init_params()

    def _to_device(self, x):
        return torch.tensor(x, device=self.device)

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (
                1,
                self.n_components,
                self.n_features,
            ), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
                self.n_components,
                self.n_features,
            )
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=True)
            self.mu = self._to_device(self.mu)
        else:
            self.mu = torch.nn.Parameter(
                torch.randn(1, self.n_components, self.n_features, device=self.device),
                requires_grad=True,
            )

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (
                    1,
                    self.n_components,
                    self.n_features,
                ), (
                    "Input var_init does not have required tensor dimensions (1, %i, %i)"
                    % (self.n_components, self.n_features)
                )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=True)
                self.var = self._to_device(self.var)
            else:
                self.var = torch.nn.Parameter(
                    torch.ones(
                        1, self.n_components, self.n_features, device=self.device
                    ),
                    requires_grad=True,
                )
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (
                    1,
                    self.n_components,
                    self.n_features,
                    self.n_features,
                ), (
                    "Input var_init does not have required tensor dimensions (1, %i, %i, %i)"
                    % (self.n_components, self.n_features, self.n_features)
                )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
                self.var = self._to_device(self.var)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features)
                    .reshape(1, 1, self.n_features, self.n_features)
                    .repeat(1, self.n_components, 1, 1),
                    requires_grad=True,
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(
            torch.Tensor(1, self.n_components, 1, device=self.device).fill_(
                1.0 / self.n_components
            ),
            requires_grad=True,
        )

        self.params = [self.pi, self.mu, self.var]
        self.fitted = False

    def _finish_optimization(self):
        self.mu_chached = copy(self.mu)
        self.var_chached = copy(self.var)
        self.fitted = True

    def _set_full(self):
        self._set_marginal(indices=[])

    def _set_marginal(self, indices=[]):
        if len(indices) == 0:
            self.mu.data = self.mu_chached.data
            self.var.data = self.var_chached.data

        else:
            max_dimension = self.mu_chached.shape[-1]
            assert any(
                [~(idx <= max_dimension) for idx in indices]
            ), f"One of provided indices {indices} is higher than a number of dimensions the model was fitted on {max_dimension}."

            self.mu.data = torch.zeros(1, self.n_components, len(indices))
            for i, ii in enumerate(indices):
                self.mu.data[:, :, i] = self.mu_chached[:, :, ii]

            if self.covariance_type is "full":
                self.var.data = torch.zeros(
                    1, self.n_components, len(indices), len(indices)
                )

                for i, ii in enumerate(indices):
                    for j, jj in enumerate(indices):
                        self.var.data[:, :, i, j] = self.var_chached[:, :, ii, jj]
            else:
                self.var.data = torch.zeros(1, self.n_components, len(indices))
                for i, ii in enumerate(indices):
                    self.mu_chached.data[:, :, i] = self.var[:, :, ii]

    def check_size(self, x):
        if len(x.size()) <= 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = (
            self.n_features * self.n_components
            + self.n_features
            + self.n_components
            - 1
        )

        bic = -2.0 * self.__score(
            x, as_average=False
        ).mean() * n + free_params * np.log(n)

        return bic

    def fit_em(self, x, delta=1e-5, n_iter=300, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_means == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)
            self.print_verbose(f"score {self.log_likelihood.item()}")
            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(
                self.log_likelihood
            ):

                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(
                    self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps,
                )
                for p in self.parameters():
                    p.data = self._to_device(p.data)
                if self.init_means == "kmeans":
                    (self.mu.data,) = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self._finish_optimization()

    def fit_grad(self, x, n_iter=1000, learning_rate=1e-1):

        # TODO make sure constrains for self.var & self.pi are satisfied
        # TODO e.g: https://www.kernel-operations.io/keops/_auto_tutorials/gaussian_mixture/plot_gaussian_mixture.html

        if self.init_means == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        optimizer = torch.optim.Adam([self.pi, self.mu, self.var], lr=learning_rate)

        # Initialise the minimum loss at infinity.
        x = self._to_device(x)
        # Iterate over the number of iterations.
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = -self.__score(x)
            loss.backward()
            self.print_verbose(f"score {loss.item()}")
            optimizer.step()

        self._finish_optimization()

    def predict(self, x, probs=False, marginals=[]):
        with MarginalContext(self, marginals) as MC:
            """
            Assigns input data to one of the mixture components by evaluating the likelihood under each.
            If probs=True returns normalized probabilities of class membership.
            args:
                x:          torch.Tensor (n, d) or (n, 1, d)
                probs:      bool
            returns:
                p_k:        torch.Tensor (n, k)
                (or)
                y:          torch.LongTensor (n)
            """
            x = self.check_size(x)

            weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
            if probs:
                p_k = torch.exp(weighted_log_prob)
                return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
            else:
                return torch.squeeze(
                    torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor)
                )

    def predict_proba(self, x, marginals=[]):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True, marignals=marginals)

    def sample(self, n, marginals=[]):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
            marginals:  empty list for all dimensions
        """
        with MarginalContext(self, marginals) as MC:
            if self.n_components == 1:
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(
                    self.mu[0, 0], self.var[0, 0]
                )
                return torch.stack([d_k.sample() for _ in range(n)]), torch.ones(
                    size=(n, 1)
                ).squeeze(1)

            else:
                counts = torch.distributions.multinomial.Multinomial(
                    total_count=n, probs=self.pi.squeeze()
                ).sample()
                x = torch.empty(0, device=counts.device)
                y = torch.cat(
                    [
                        torch.full([int(sample)], j, device=counts.device)
                        for j, sample in enumerate(counts)
                    ]
                )

                # Only iterate over components with non-zero counts
                for k in np.arange(self.n_components)[counts > 0]:
                    if self.covariance_type == "diag":
                        x_k = self.mu[0, k] + torch.randn(
                            int(counts[k]), self.n_features, device=x.device
                        ) * torch.sqrt(self.var[0, k])
                    elif self.covariance_type == "full":
                        d_k = (
                            torch.distributions.multivariate_normal.MultivariateNormal(
                                self.mu[0, k], self.var[0, k]
                            )
                        )
                        x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

                    x = torch.cat((x, x_k), dim=0)

        return x, y

    def logscore_samples(self, x, marginals=[]):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            marginals:  empty list for all dimensions
        returns:
            score:      log_scores: torch.LongTensor (n)
        """

        with MarginalContext(self, marginals) as MC:
            x = self.check_size(x)
            score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            if var.shape[2] == 1:
                precision = 1 / var
            else:
                precision = torch.inverse(var)

            d = x.shape[-1]

            log_2pi = d * np.log(2.0 * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(
                self.n_components, x_mu_T, precision
            )
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -0.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum(
                (mu * mu + x * x - 2 * x * mu) * (prec**2), dim=2, keepdim=True
            )
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -0.5 * (self.n_features * np.log(2.0 * pi) + log_p) + log_det

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = (
                2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()
            )

        return log_det.unsqueeze(-1)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = (
                torch.sum(
                    (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
                    * resp.unsqueeze(-1),
                    dim=0,
                    keepdim=True,
                )
                / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1)
                + eps
            )
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [
            (self.n_components, self.n_features),
            (1, self.n_components, self.n_features),
        ], (
            "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
            % (self.n_components, self.n_features, self.n_components, self.n_features)
        )

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [
                (self.n_components, self.n_features, self.n_features),
                (1, self.n_components, self.n_features, self.n_features),
            ], (
                "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)"
                % (
                    self.n_components,
                    self.n_features,
                    self.n_features,
                    self.n_components,
                    self.n_features,
                    self.n_features,
                )
            )

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [
                (self.n_components, self.n_features),
                (1, self.n_components, self.n_features),
            ], (
                "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
                % (
                    self.n_components,
                    self.n_features,
                    self.n_components,
                    self.n_features,
                )
            )

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [
            (1, self.n_components, 1)
        ], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1,
            self.n_components,
            1,
        )

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[
                np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False),
                ...,
            ]
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2
            )
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2
            )
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return self._to_device(center.unsqueeze(0) * (x_max - x_min) + x_min)

    def print_verbose(self, string):
        if self.verbose:
            print(string)
