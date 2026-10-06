"""
Bayesian logistic regression by mean-field stochastic variational inference.

    training    local reparameterisation trick: the pre-activation is sampled
                per example.
    prediction  one global weight vector is drawn from the variational posterior
                and applied to the whole split, repeated `num_samples` times, so
                the spread across draws measures disagreement between models.

Priors (`bnn.prior`):

    scale_mixture    pi N(0, sigma1^2) + (1 - pi) N(0, sigma2^2); KL is estimated
                     with one weight sample per step.
    scaled_gaussian  N(0, s^2), s = prior_scale or 1/sqrt(fan_in); closed-form KL.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import log_elapsed

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- priors
class WeightPrior:
    """Base prior: exposes either a closed-form KL or a log-density for MC KL."""

    closed_form_kl = False

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def kl(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ScaledGaussianPrior(WeightPrior):
    """N(0, s^2) with s fixed per layer; KL from a Gaussian posterior is analytic."""

    scale: float
    closed_form_kl = True

    def log_prob(self, w):
        return (
            -0.5 * (w / self.scale) ** 2
            - math.log(self.scale)
            - 0.5 * math.log(2 * math.pi)
        ).sum()

    def kl(self, mu, sigma):
        s = self.scale
        return (
            torch.log(torch.tensor(s, device=mu.device) / sigma)
            + (sigma**2 + mu**2) / (2 * s**2)
            - 0.5
        ).sum()


@dataclass
class ScaleMixturePrior(WeightPrior):
    """pi N(0, sigma1^2) + (1 - pi) N(0, sigma2^2); sparsity-inducing, MC KL."""

    pi: float
    sigma1: float
    sigma2: float
    closed_form_kl = False

    def log_prob(self, w):
        def comp(sigma):
            return (
                -0.5 * (w / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            )

        stacked = torch.stack(
            [
                comp(self.sigma1) + math.log(self.pi),
                comp(self.sigma2) + math.log(1 - self.pi),
            ]
        )
        return torch.logsumexp(stacked, dim=0).sum()


def build_prior(cfg, fan_in: int) -> WeightPrior:
    """Instantiate the configured prior for a layer with `fan_in` inputs."""
    if cfg.prior == "scaled_gaussian":
        scale = cfg.prior_scale if cfg.prior_scale else 1.0 / math.sqrt(fan_in)
        return ScaledGaussianPrior(scale=float(scale))
    if cfg.prior == "scale_mixture":
        return ScaleMixturePrior(
            pi=cfg.mixture_pi, sigma1=cfg.mixture_sigma1, sigma2=cfg.mixture_sigma2
        )
    raise ValueError(f"unknown bnn.prior '{cfg.prior}'")


# -------------------------------------------------------------------- layer
class BayesianLinear(nn.Module):
    """
    Linear layer with a fully factorised Gaussian posterior over weights and bias.

    `forward` is used only in training (local reparameterisation); prediction
    goes through `sample_weights` / `forward_with_weights`.
    """

    def __init__(self, in_features: int, out_features: int, prior: WeightPrior, init_rho: float = -5.0):
        super().__init__()
        self.in_features, self.out_features, self.prior = in_features, out_features, prior

        # Uniform init on the mean; small rho so the posterior starts tight
        bound = 1.0 / math.sqrt(in_features)
        self.w_mu = nn.Parameter(torch.empty(in_features, out_features).uniform_(-bound, bound))
        self.w_rho = nn.Parameter(torch.full((in_features, out_features), init_rho))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), init_rho))

    @property
    def w_sigma(self) -> torch.Tensor:
        return F.softplus(self.w_rho)

    @property
    def b_sigma(self) -> torch.Tensor:
        return F.softplus(self.b_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_lrt(x)
        raise RuntimeError(
            "BayesianLinear.forward was called in eval mode. Prediction must go "
            "through sample_weights()/forward_with_weights() so that a single "
            "global weight draw is shared across all patients."
        )

    def _forward_lrt(self, x: torch.Tensor) -> torch.Tensor:
        """Sample the pre-activation directly: a ~ N(xM, x^2 S^2)."""
        act_mu = x @ self.w_mu + self.b_mu
        x_sq = x * x
        act_var = x_sq @ (self.w_sigma**2) + self.b_sigma**2
        return act_mu + torch.sqrt(act_var + 1e-12) * torch.randn_like(act_mu)

    def sample_weights(self, generator: torch.Generator | None = None):
        """One global (W, b) draw from the variational posterior."""
        eps_w = torch.randn(self.w_mu.shape, device=self.w_mu.device, generator=generator)
        eps_b = torch.randn(self.b_mu.shape, device=self.b_mu.device, generator=generator)
        return self.w_mu + self.w_sigma * eps_w, self.b_mu + self.b_sigma * eps_b

    @staticmethod
    def forward_with_weights(x, w, b):
        return x @ w + b

    def kl(self) -> torch.Tensor:
        """KL(q || prior): closed form for the Gaussian prior, one-sample MC otherwise."""
        if self.prior.closed_form_kl:
            return self.prior.kl(self.w_mu, self.w_sigma) + self.prior.kl(self.b_mu, self.b_sigma)
        w, b = self.sample_weights()
        return (_gaussian_log_prob(w, self.w_mu, self.w_sigma) - self.prior.log_prob(w)) + (
            _gaussian_log_prob(b, self.b_mu, self.b_sigma) - self.prior.log_prob(b)
        )

    def sparsity(self, threshold: float = 0.1) -> float:
        """Fraction of weights whose posterior mean is small relative to its SD."""
        with torch.no_grad():
            return float((self.w_mu.abs() / (self.w_sigma + 1e-12) < threshold).float().mean())


def _gaussian_log_prob(x, mu, sigma):
    return (-0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum()


# -------------------------------------------------------------------- model
class BayesianMLP(nn.Module):
    """Stack of Bayesian linear layers; `dims=[d, 1]` is Bayesian logistic regression."""

    def __init__(self, dims: list[int], cfg):
        super().__init__()
        if len(dims) < 2 or dims[-1] != 1:
            raise ValueError("dims must have >= 2 entries and end in 1")
        self.layers = nn.ModuleList(
            [
                BayesianLinear(dims[i], dims[i + 1], build_prior(cfg, dims[i]), cfg.init_rho)
                for i in range(len(dims) - 1)
            ]
        )
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time forward using LRT at every layer."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x.squeeze(-1)

    def forward_sampled(self, x: torch.Tensor, weights: list) -> torch.Tensor:
        """Evaluation-time forward using one shared global weight draw."""
        for i, (layer, (w, b)) in enumerate(zip(self.layers, weights)):
            x = layer.forward_with_weights(x, w, b)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x.squeeze(-1)

    def sample_all_weights(self, generator=None) -> list:
        return [layer.sample_weights(generator) for layer in self.layers]

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ dataset
class _Batches:
    """Minibatch iterator over a dense array or a scipy sparse matrix."""

    def __init__(self, X, y, batch_size: int, rng: np.random.Generator, device):
        self.X, self.y, self.batch_size, self.rng, self.device = X, y, batch_size, rng, device
        self.n = X.shape[0]
        self.sparse = hasattr(X, "toarray")

    def __iter__(self):
        for start in range(0, self.n, self.batch_size):
            idx = self.order[start : start + self.batch_size]
            chunk = self.X[idx]
            dense = chunk.toarray() if self.sparse else chunk
            yield (
                torch.as_tensor(np.ascontiguousarray(dense), dtype=torch.float32, device=self.device),
                torch.as_tensor(self.y[idx], dtype=torch.float32, device=self.device),
            )

    def epoch(self):
        self.order = self.rng.permutation(self.n)
        return self

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))


# ------------------------------------------------------------------ trainer
def build_network(cfg, n_features: int) -> BayesianMLP:
    """Bayesian logistic regression over all `n_features` inputs."""
    dims = [n_features, 1]
    net = BayesianMLP(dims, cfg)
    logger.info(
        "BLR dims=%s prior=%s variational parameters=%d",
        dims,
        cfg.prior,
        net.n_parameters,
    )
    return net


def fit_bnn(X_train, y_train, cfg, seed: int = 0) -> BayesianMLP:
    """
    Fit the variational posterior with Adam on the negative ELBO.

    KL is scaled by batch_size / N and annealed linearly over the first
    `kl_warmup_frac` of training steps.
    """
    device = torch.device(cfg.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n, d = X_train.shape
    net = build_network(cfg, d).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    batches = _Batches(X_train, np.asarray(y_train), cfg.batch_size, rng, device)

    total_steps = cfg.epochs * len(batches)
    warmup_steps = max(1, int(cfg.kl_warmup_frac * total_steps))
    step = 0

    net.train()
    with log_elapsed(f"BLR training ({cfg.epochs} epochs)"):
        for epoch in range(1, cfg.epochs + 1):
            nll_sum = kl_sum = 0.0
            for xb, yb in batches.epoch():
                beta = min(1.0, step / warmup_steps)
                logits = net(xb)
                nll = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum")
                kl = net.kl() * (len(yb) / n)
                loss = nll + beta * kl

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                optimiser.step()

                nll_sum += float(nll.detach())
                kl_sum += float(kl.detach())
                step += 1

            logger.info(
                "  epoch %d/%d | NLL/n=%.4f  KL=%.1f  beta=%.2f",
                epoch, cfg.epochs, nll_sum / n, kl_sum, beta,
            )

    for i, layer in enumerate(net.layers):
        logger.info("  layer %d posterior sparsity (|mu|/sigma < 0.1): %.3f", i, layer.sparsity())
    return net


@torch.no_grad()
def predict_bnn(net: BayesianMLP, X, cfg, seed: int = 0) -> np.ndarray:
    """Posterior predictive probabilities, shape (num_samples, n_eval); one global weight draw per row."""
    device = torch.device(cfg.device)
    net.eval().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    sparse = hasattr(X, "toarray")
    n = X.shape[0]
    out = np.empty((cfg.num_samples, n), dtype=np.float64)

    with log_elapsed(f"BLR posterior predictive ({cfg.num_samples} draws)"):
        for s in range(cfg.num_samples):
            weights = net.sample_all_weights(generator)
            for start in range(0, n, cfg.predict_batch_size):
                stop = min(start + cfg.predict_batch_size, n)
                chunk = X[start:stop]
                dense = chunk.toarray() if sparse else chunk
                xb = torch.as_tensor(
                    np.ascontiguousarray(dense), dtype=torch.float32, device=device
                )
                logits = net.forward_sampled(xb, weights)
                out[s, start:stop] = torch.sigmoid(logits).cpu().numpy()
    return out


def check_convergence(probs: np.ndarray, y_true, cfg, split: str = "validation") -> float:
    """Log validation AUROC and an error if it falls below `bnn.min_val_auroc`."""
    from sklearn.metrics import roc_auc_score

    mean_proba = probs.mean(axis=0)
    y = np.asarray(y_true, dtype=int)
    auroc = float(roc_auc_score(y, mean_proba)) if len(np.unique(y)) > 1 else float("nan")
    logger.info("BLR %s AUROC=%.4f  mean p-bar=%.4f", split, auroc, float(mean_proba.mean()))

    if np.isfinite(auroc) and auroc < cfg.min_val_auroc:
        logger.error(
            "BLR %s AUROC=%.3f is below min_val_auroc=%.3f; the posterior may have "
            "collapsed onto the prior.",
            split, auroc, cfg.min_val_auroc,
        )
    return auroc
