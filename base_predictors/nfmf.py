"""Joint RT/TP NFMF with context, identity embeddings, and task-specific heads.

Missing categorical values receive a reserved category and missing coordinates
use the standardised training mean. Raw-target L1 is the reproduction default;
standardised targets are available only as an explicit alternative."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EMBEDDING_DIM = 64
TASK_LAYERS = (64, 32, 1)
CONTEXT_LAYERS = (64, 64, 64)
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCHS = 200
EARLY_STOPPING_ROUNDS = 15
BIAS_WEIGHT = 1.0
TASKS = ("rt", "tp")
# Categorical context, in the order they are concatenated. Four continuous
# columns -- both latitudes and both longitudes -- follow, giving 7*64 + 4.
USER_FIELDS = ("u_as", "u_country", "u_ip_class")
SERVICE_FIELDS = ("s_as", "s_country", "s_provider", "s_ip_class")
CONTEXT_FIELDS = USER_FIELDS + SERVICE_FIELDS
# First octet to address class, as the paper does. Anything outside the five
# ranges -- 127, and rows with no address at all -- gets its own level rather
# than being forced into one of them.
IP_CLASS_RANGES = ((0, 126, "A"), (128, 191, "B"), (192, 223, "C"),
                   (224, 239, "D"), (240, 255, "E"))


def ip_class(address) -> str:
    try:
        octet = int(str(address).split(".")[0])
    except (ValueError, AttributeError):
        return "missing"
    for low, high, name in IP_CLASS_RANGES:
        if low <= octet <= high:
            return name
    return "other"


def _codes(values: pd.Series) -> tuple[np.ndarray, int]:
    """Integer codes with missing values kept as a level of their own."""
    filled = values.astype("object").where(values.notna(), "__missing__")
    category = pd.Categorical(filled.astype(str))
    return category.codes.astype(np.int64), len(category.categories)


def build_context(users: pd.DataFrame, services: pd.DataFrame) -> dict:
    """Build context tables shared across splits, densities, and seeds."""
    raw = {
        "u_as": users.as_no, "u_country": users.country,
        "u_ip_class": pd.Series([ip_class(v) for v in users.ip], index=users.index),
        "s_as": services.as_no, "s_country": services.country,
        "s_provider": services.provider,
        "s_ip_class": pd.Series([ip_class(v) for v in services.ip], index=services.index),
    }
    codes, sizes = {}, {}
    for field, series in raw.items():
        codes[field], sizes[field] = _codes(series)
    numeric = {
        "u_lat": users.lat.to_numpy(float), "u_lon": users.lon.to_numpy(float),
        "s_lat": services.lat.to_numpy(float), "s_lon": services.lon.to_numpy(float),
    }
    return {"codes": codes, "sizes": sizes, "numeric": numeric,
            "n_users": len(users), "n_services": len(services)}


@dataclass(frozen=True)
class NfmfHistoryRow:
    epoch: int
    train_loss: float
    validation_mae_rt: float | None
    validation_mae_tp: float | None
    validation_score: float | None


def _tower(width: int, layers: tuple[int, ...]) -> nn.Sequential:
    """{width, *layers}: ReLU between, plain linear out."""
    stack: list[nn.Module] = []
    for size in layers[:-1]:
        stack += [nn.Linear(width, size), nn.ReLU()]
        width = size
    stack.append(nn.Linear(width, layers[-1]))
    return nn.Sequential(*stack)


class ContextEncoder(nn.Module):
    """One embedding table per categorical field, then a batch-normalised MLP."""

    def __init__(self, sizes: dict, dimension: int = EMBEDDING_DIM,
                 layers: tuple[int, ...] = CONTEXT_LAYERS, n_numeric: int = 4):
        super().__init__()
        self.tables = nn.ModuleDict(
            {field: nn.Embedding(sizes[field], dimension) for field in CONTEXT_FIELDS})
        width = dimension * len(CONTEXT_FIELDS) + n_numeric
        self.input_size = width
        stack: list[nn.Module] = []
        for size in layers:
            stack += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU()]
            width = size
        self.mlp = nn.Sequential(*stack)

    def forward(self, codes: dict, numeric: torch.Tensor) -> torch.Tensor:
        parts = [self.tables[field](codes[field]) for field in CONTEXT_FIELDS]
        return self.mlp(torch.cat(parts + [numeric], dim=1))


class TaskBranch(nn.Module):
    """NFCF and bias networks for one QoS type."""

    def __init__(self, dimension: int = EMBEDDING_DIM,
                 layers: tuple[int, ...] = TASK_LAYERS,
                 use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        # [p; q; c] is 3d and p+q+c and p*q*c are d each, so 5d in total; with
        # no context vector the same three parts come to 4d.
        self.nfcf = _tower((5 if use_context else 4) * dimension, layers)
        width = (2 if use_context else 1) * dimension
        self.user_bias = _tower(width, layers)
        self.service_bias = _tower(width, layers)

    def forward(self, p: torch.Tensor, q: torch.Tensor,
                c: torch.Tensor | None, bias_weight: float) -> torch.Tensor:
        if self.use_context:
            joined = torch.cat([p, q, c, p + q + c, p * q * c], dim=1)
            bias = (self.user_bias(torch.cat([p, c], dim=1))
                    + self.service_bias(torch.cat([q, c], dim=1)))
        else:
            joined = torch.cat([p, q, p + q, p * q], dim=1)
            bias = self.user_bias(p) + self.service_bias(q)
        return (self.nfcf(joined) + bias_weight * bias).squeeze(1)


class NfmfNet(nn.Module):
    def __init__(self, n_users: int, n_services: int, sizes: dict,
                 dimension: int = EMBEDDING_DIM,
                 layers: tuple[int, ...] = TASK_LAYERS,
                 context_layers: tuple[int, ...] = CONTEXT_LAYERS,
                 bias_weight: float = BIAS_WEIGHT,
                 use_context: bool = True):
        super().__init__()
        self.bias_weight = bias_weight
        self.use_context = use_context
        self.user = nn.Embedding(n_users, dimension)
        self.service = nn.Embedding(n_services, dimension)
        self.context = (ContextEncoder(sizes, dimension, context_layers)
                        if use_context else None)
        self.branches = nn.ModuleDict(
            {task: TaskBranch(dimension, layers, use_context) for task in TASKS})

    def forward(self, users: torch.Tensor, services: torch.Tensor,
                codes: dict, numeric: torch.Tensor | None) -> dict:
        p, q = self.user(users), self.service(services)
        c = self.context(codes, numeric) if self.use_context else None
        return {task: branch(p, q, c, self.bias_weight)
                for task, branch in self.branches.items()}


class NfmfPredictor:
    """Fit RT and TP jointly; disabled validation uses the full epoch budget."""

    def __init__(self, context: dict,
                 dimension: int = EMBEDDING_DIM,
                 layers: tuple[int, ...] = TASK_LAYERS,
                 context_layers: tuple[int, ...] = CONTEXT_LAYERS,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
                 early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                 bias_weight: float = BIAS_WEIGHT,
                 target_scaling: str = "none",
                 validation_fraction: float = 0.1,
                 seed: int = 42, init_seed: int | None = None,
                 threads: int = 4, device: str | None = None,
                 use_context: bool = True):
        if target_scaling not in ("none", "standardise"):
            raise ValueError(f"target_scaling: {target_scaling!r}")
        self.context, self.target_scaling = context, target_scaling
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        # The split draw and the weight draw are kept apart so a run can be
        # repeated with one held and the other moved.
        self.seed = seed
        self.init_seed = seed if init_seed is None else init_seed
        self.history: list[NfmfHistoryRow] = []
        self.best_epoch: int | None = None
        self.scale = {task: (0.0, 1.0) for task in TASKS}
        self.numeric_scale: tuple[np.ndarray, np.ndarray] | None = None

        random.seed(self.init_seed)
        np.random.seed(self.init_seed)
        torch.manual_seed(self.init_seed)
        torch.cuda.manual_seed_all(self.init_seed)
        torch.set_num_threads(threads)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_context = use_context
        self.model = NfmfNet(context["n_users"], context["n_services"],
                             context["sizes"], dimension, layers,
                             context_layers, bias_weight,
                             use_context).to(self.device)

    # ---- inputs -----------------------------------------------------------

    def _codes(self, entries: np.ndarray) -> dict:
        users = entries[:, 0].astype(np.int64)
        services = entries[:, 1].astype(np.int64)
        index = {field: (users if field.startswith("u_") else services)
                 for field in CONTEXT_FIELDS}
        return {field: torch.as_tensor(self.context["codes"][field][index[field]],
                                       device=self.device)
                for field in CONTEXT_FIELDS}

    def _numeric(self, entries: np.ndarray) -> torch.Tensor:
        users = entries[:, 0].astype(np.int64)
        services = entries[:, 1].astype(np.int64)
        columns = [self.context["numeric"]["u_lat"][users],
                   self.context["numeric"]["u_lon"][users],
                   self.context["numeric"]["s_lat"][services],
                   self.context["numeric"]["s_lon"][services]]
        raw = np.stack(columns, axis=1)
        mean, std = self.numeric_scale
        # Rows with no coordinates take the sub-training mean, which is zero
        # once standardised: the encoder then sees no displacement from them.
        filled = np.where(np.isnan(raw), mean, raw)
        return torch.as_tensor(((filled - mean) / std).astype(np.float32),
                               device=self.device)

    # ---- fitting ----------------------------------------------------------

    def fit(self, entries: np.ndarray, targets: dict) -> "NfmfPredictor":
        missing = set(TASKS) - set(targets)
        if missing:
            raise KeyError(f"targets에 {sorted(missing)} 없음")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(entries))
        cut = int(round(len(entries) * (1.0 - self.validation_fraction)))
        fit_idx, val_idx = order[:cut], order[cut:]

        # Every statistic comes from the fitting rows alone, coordinates
        # included, so the held-out tenth touches neither the weights nor any
        # scale they are expressed in.
        sub = entries[fit_idx]
        users, services = sub[:, 0].astype(np.int64), sub[:, 1].astype(np.int64)
        raw = np.stack([self.context["numeric"]["u_lat"][users],
                        self.context["numeric"]["u_lon"][users],
                        self.context["numeric"]["s_lat"][services],
                        self.context["numeric"]["s_lon"][services]], axis=1)
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        self.numeric_scale = (mean, np.where(std > 0, std, 1.0))

        values = {task: np.asarray(targets[task], dtype=np.float64) for task in TASKS}
        if self.target_scaling == "standardise":
            self.scale = {task: (float(v[fit_idx].mean()),
                                 float(v[fit_idx].std()) or 1.0)
                          for task, v in values.items()}
        scaled = {task: torch.as_tensor(
            ((v - self.scale[task][0]) / self.scale[task][1]).astype(np.float32),
            device=self.device)
            for task, v in values.items()}

        all_users = torch.as_tensor(entries[:, 0].astype(np.int64), device=self.device)
        all_services = torch.as_tensor(entries[:, 1].astype(np.int64), device=self.device)
        # Neither is built when the encoder is gone: the side information is
        # not read at all in that condition.
        codes = self._codes(entries) if self.use_context else {}
        numeric = self._numeric(entries) if self.use_context else None
        # The split keeps only entries valid on both QoS types, so the two
        # masks coincide; the loss is still written per task so the shape of it
        # does not depend on that staying true.
        mask = {task: torch.as_tensor(np.isfinite(values[task]), device=self.device)
                for task in TASKS}

        fit_t = torch.as_tensor(fit_idx, device=self.device)
        val_t = (torch.as_tensor(val_idx, device=self.device) if len(val_idx)
                 else None)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        generator = torch.Generator().manual_seed(self.init_seed)
        best_score, best_state, stale = np.inf, None, 0

        def losses(index: torch.Tensor) -> dict:
            out = self.model(all_users[index], all_services[index],
                             {f: c[index] for f, c in codes.items()},
                             None if numeric is None else numeric[index])
            return {task: torch.abs(out[task][mask[task][index]]
                                    - scaled[task][index][mask[task][index]]).mean()
                    for task in TASKS}

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            # The permutation is drawn on the CPU so the order does not depend
            # on which device the run happens to use.
            shuffled = fit_t[torch.randperm(len(fit_t), generator=generator)
                             .to(self.device)]
            total = 0.0
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) < 2:
                    continue  # batch normalisation needs more than one row
                optimiser.zero_grad()
                parts = losses(batch)
                loss = sum(parts.values()) / len(TASKS)
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(batch)

            maes, score = {task: None for task in TASKS}, None
            if val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    parts = losses(val_t)
                maes = {task: float(value) for task, value in parts.items()}
                score = sum(maes.values()) / len(TASKS)
                if score + 1e-9 < best_score:
                    best_score, self.best_epoch, stale = score, epoch, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    stale += 1
            self.history.append(NfmfHistoryRow(
                epoch, total / len(fit_t), maes["rt"], maes["tp"], score))
            if val_t is not None and stale >= self.early_stopping_rounds:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch = len(self.history)
        return self

    @torch.no_grad()
    def predict_entries(self, entries: np.ndarray,
                        batch_size: int = 65536) -> dict:
        """Both QoS types from the one model. No floor at zero: the task
        networks end in a plain linear layer, as the published ones do."""
        self.model.eval()
        users = torch.as_tensor(entries[:, 0].astype(np.int64), device=self.device)
        services = torch.as_tensor(entries[:, 1].astype(np.int64), device=self.device)
        codes = self._codes(entries) if self.use_context else {}
        numeric = self._numeric(entries) if self.use_context else None
        out = {task: np.empty(len(entries), dtype=np.float64) for task in TASKS}
        for start in range(0, len(entries), batch_size):
            stop = start + batch_size
            block = self.model(users[start:stop], services[start:stop],
                               {f: c[start:stop] for f, c in codes.items()},
                               None if numeric is None else numeric[start:stop])
            for task in TASKS:
                mean, std = self.scale[task]
                out[task][start:stop] = (block[task].cpu().numpy().astype(np.float64)
                                     * std + mean)
        return out
