"""LDCF reimplementation based on the authors' released code.

Source: yinchunhui-ahu/Location-aware_Deep_Collaborative_Filtering, commit
3934e06. The original shared embedding tables and their published sizing are
preserved for faithful comparison."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# `parse_args` defaults at commit 3934e06.
LAYERS = (64, 32, 16, 8, 1)
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
EPOCHS = 50
HUBER_DELTA = 1.0
# Keras `initializers.random_normal()` is N(0, 0.05); the output layer uses
# `lecun_normal()`, which is N(0, sqrt(1/fan_in)).
INIT_STD = 0.05


@dataclass(frozen=True)
class LdcfHistoryRow:
    epoch: int
    train_loss: float
    validation_rmse: float | None
    validation_mae: float | None


class LdcfNet(nn.Module):
    def __init__(self, n_users: int, n_services: int,
                 layers: tuple[int, ...] = LAYERS):
        super().__init__()
        dim = layers[0] // 4
        self.user_id = nn.Embedding(n_users, dim)
        self.user_lc = nn.Embedding(n_users, dim)
        self.item_id = nn.Embedding(n_services, dim)
        self.item_lc = nn.Embedding(n_services, dim)
        for embedding in (self.user_id, self.user_lc, self.item_id, self.item_lc):
            nn.init.normal_(embedding.weight, std=INIT_STD)

        # id (dim) + two location slots (2*dim) on each side.
        width = 6 * dim
        blocks: list[nn.Module] = []
        for out in layers[1:-1]:
            linear = nn.Linear(width, out)
            nn.init.normal_(linear.weight, std=INIT_STD)
            nn.init.zeros_(linear.bias)
            blocks += [linear, nn.ReLU()]
            width = out
        self.mlp = nn.Sequential(*blocks)

        # The cosine scalar joins the MLP output just before the linear head.
        self.head = nn.Linear(width + 1, layers[-1])
        nn.init.normal_(self.head.weight, std=math.sqrt(1.0 / (width + 1)))
        nn.init.zeros_(self.head.bias)

    def forward(self, users: torch.Tensor, user_lc: torch.Tensor,
                services: torch.Tensor, item_lc: torch.Tensor) -> torch.Tensor:
        u_id = self.user_id(users)
        u_lc = self.user_lc(user_lc).flatten(1)
        i_id = self.item_id(services)
        i_lc = self.item_lc(item_lc).flatten(1)
        hidden = self.mlp(torch.cat([u_id, u_lc, i_id, i_lc], dim=1))
        # AC-COS: Keras `dot(..., normalize=True)` on the two location vectors.
        cosine = F.cosine_similarity(u_lc, i_lc, dim=1, eps=1e-12).unsqueeze(1)
        return self.head(torch.cat([hidden, cosine], dim=1)).squeeze(1)


class LdcfPredictor:
    """Train LDCF with Adam and Huber loss using validation-based epoch selection."""

    def __init__(self, n_users: int, n_services: int,
                 user_lc: np.ndarray, item_lc: np.ndarray,
                 layers: tuple[int, ...] = LAYERS,
                 learning_rate: float = LEARNING_RATE,
                 batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
                 validation_fraction: float = 0.1, seed: int = 42,
                 threads: int = 4):
        self.layers, self.epochs = layers, epochs
        self.learning_rate, self.batch_size = learning_rate, batch_size
        self.validation_fraction, self.seed = validation_fraction, seed
        self.user_lc, self.item_lc = user_lc, item_lc
        self.history: list[LdcfHistoryRow] = []
        self.best_epoch: int | None = None
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        self.model = LdcfNet(n_users, n_services, layers)

    def _tensors(self, entries: np.ndarray):
        users = entries[:, 0].astype(np.int64)
        services = entries[:, 1].astype(np.int64)
        return (torch.as_tensor(users), torch.as_tensor(self.user_lc[users]),
                torch.as_tensor(services), torch.as_tensor(self.item_lc[services]))

    def fit(self, entries: np.ndarray, values: np.ndarray) -> "LdcfPredictor":
        """Use all entries for the full budget when validation is disabled."""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(entries))
        cut = int(round(len(entries) * (1.0 - self.validation_fraction)))
        fit_idx, val_idx = order[:cut], order[cut:]

        inputs = self._tensors(entries[fit_idx])
        target = torch.as_tensor(values[fit_idx].astype(np.float32))
        val_inputs = self._tensors(entries[val_idx]) if len(val_idx) else None
        val_true = values[val_idx].astype(np.float64)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss(delta=HUBER_DELTA)
        generator = torch.Generator().manual_seed(self.seed)
        best_mae, best_state = np.inf, None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            shuffled = torch.randperm(len(target), generator=generator)
            total = 0.0
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.model(*[t[batch] for t in inputs]), target[batch])
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(batch)

            rmse = mae = None
            if val_inputs is not None:
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model(*val_inputs).numpy().astype(np.float64)
                residual = val_true - pred
                mae = float(np.mean(np.abs(residual)))
                rmse = float(np.sqrt(np.mean(residual ** 2)))
                # MAE, not RMSE: the original tracks MAE and the paper reports it.
                if mae < best_mae:
                    best_mae, self.best_epoch = mae, epoch
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
            self.history.append(LdcfHistoryRow(epoch, total / len(target), rmse, mae))

        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch = self.epochs
        return self

    @torch.no_grad()
    def predict_entries(self, entries: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """No floor at zero: the original's output layer is plain linear."""
        self.model.eval()
        out = np.empty(len(entries), dtype=np.float64)
        for start in range(0, len(entries), batch_size):
            block = entries[start:start + batch_size]
            out[start:start + len(block)] = self.model(*self._tensors(block)).numpy()
        return out
