"""PMF and BiasMF as implemented by LanternYing/WSRec.

Both variants use the published summed-error objective, full-batch gradient
descent with line search, and unclamped predictions. ``use_bias`` selects the
BiasMF variant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


# From `PMF/src/core/PMF.cpp: linesearch` -- the trial step is halved at most
# this many times before the iteration gives up and takes the last one.
LINE_SEARCH_STEPS = 20


@dataclass(frozen=True)
class WsrecHistoryRow:
    iteration: int
    train_loss: float
    eta: float
    validation_rmse: float | None
    validation_mae: float | None


class WsrecMatrixFactorization:
    """Implement PMF or BiasMF with the published uniform initialisation."""

    def __init__(self, n_users: int, n_services: int, n_factors: int = 10,
                 lam: float = 40.0, eta_init: float = 0.01, max_iter: int = 300,
                 seed: int = 42, use_bias: bool = False,
                 line_search_steps: int = LINE_SEARCH_STEPS):
        self.n_users, self.n_services, self.n_factors = n_users, n_services, n_factors
        self.lam, self.eta_init, self.max_iter = lam, eta_init, max_iter
        self.seed, self.use_bias = seed, use_bias
        self.line_search_steps = line_search_steps
        self.history: list[WsrecHistoryRow] = []
        self.mu = 0.0
        self.user_factors = self.service_factors = None
        self.user_bias = self.service_bias = None

    # -- model -------------------------------------------------------------

    def _raw(self, U, S, bu, bs, users, services) -> np.ndarray:
        return (self.mu + bu[users] + bs[services]
                + np.einsum("ij,ij->i", U[users], S[services]))

    def _loss(self, U, S, bu, bs, users, services, values) -> float:
        error = values - self._raw(U, S, bu, bs, users, services)
        penalty = float(np.sum(U * U) + np.sum(S * S))
        if self.use_bias:
            penalty += float(np.sum(bu * bu) + np.sum(bs * bs))
        return 0.5 * float(error @ error) + 0.5 * self.lam * penalty

    def _gradients(self, U, S, bu, bs, users, services, values):
        """Compute the published summed-loss gradient using sparse products."""
        error = values - self._raw(U, S, bu, bs, users, services)
        residual = csr_matrix((error, (users, services)),
                              shape=(self.n_users, self.n_services))
        grad_u = self.lam * U - residual @ S
        grad_s = self.lam * S - residual.T @ U
        if self.use_bias:
            grad_bu = self.lam * bu - np.asarray(residual.sum(axis=1)).ravel()
            grad_bs = self.lam * bs - np.asarray(residual.sum(axis=0)).ravel()
        else:
            grad_bu = np.zeros_like(bu)
            grad_bs = np.zeros_like(bs)
        return grad_u, grad_s, grad_bu, grad_bs

    # -- fitting -----------------------------------------------------------

    def fit(self, entries: np.ndarray, values: np.ndarray,
            validation_entries: np.ndarray | None = None,
            validation_values: np.ndarray | None = None) -> "WsrecMatrixFactorization":
        users = entries[:, 0].astype(np.int64)
        services = entries[:, 1].astype(np.int64)
        values = values.astype(np.float64)

        rng = np.random.default_rng(self.seed)
        U = rng.random((self.n_users, self.n_factors))
        S = rng.random((self.n_services, self.n_factors))
        bu = rng.random(self.n_users) if self.use_bias else np.zeros(self.n_users)
        bs = rng.random(self.n_services) if self.use_bias else np.zeros(self.n_services)
        # BiasMF holds the global mean at the training average; PMF has no such
        # term at all, so the prediction is the inner product alone.
        self.mu = float(values.mean()) if self.use_bias else 0.0

        last_loss = self._loss(U, S, bu, bs, users, services, values)
        for iteration in range(1, self.max_iter + 1):
            grad_u, grad_s, grad_bu, grad_bs = self._gradients(
                U, S, bu, bs, users, services, values)

            eta = self.eta_init
            for _ in range(self.line_search_steps):
                trial = self._loss(U - eta * grad_u, S - eta * grad_s,
                                   bu - eta * grad_bu, bs - eta * grad_bs,
                                   users, services, values)
                if trial <= last_loss:
                    break
                eta /= 2

            U = U - eta * grad_u
            S = S - eta * grad_s
            if self.use_bias:
                bu = bu - eta * grad_bu
                bs = bs - eta * grad_bs
            last_loss = self._loss(U, S, bu, bs, users, services, values)

            self.user_factors, self.service_factors = U, S
            self.user_bias, self.service_bias = bu, bs
            rmse = mae = None
            if validation_entries is not None and validation_values is not None:
                residual = validation_values - self.predict_entries(validation_entries)
                rmse = float(np.sqrt(np.mean(residual ** 2)))
                mae = float(np.mean(np.abs(residual)))
            self.history.append(WsrecHistoryRow(iteration, last_loss, eta, rmse, mae))

        return self

    # -- inference ---------------------------------------------------------

    def predict_entries(self, entries: np.ndarray) -> np.ndarray:
        """No floor at zero: the published code does not clamp."""
        return self._raw(self.user_factors, self.service_factors,
                         self.user_bias, self.service_bias,
                         entries[:, 0].astype(np.int64),
                         entries[:, 1].astype(np.int64))
