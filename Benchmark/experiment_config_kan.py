"""
experiment_config.py — Configuration benchmark RAN vs KAN vs MLP
=================================================================

Utilisation :
    python benchmark.py experiment_config.py
    python benchmark.py experiment_config.py --output results/ --verbose
    python benchmark.py experiment_config.py --no-plots
    python3 -m Benchmark.benchmark Benchmark/experiment_config_kan.py
"""

import torch
import torch.nn as nn
import sys, os

# ── Imports modèles ────────────────────────────────────────────────────────────
# Adapter les chemins selon votre structure de projet
sys.path.insert(0, os.path.dirname(__file__))

from Model.RecurentHyperNetwork.RecurentAttentionNetwork import RecurentAttentionNetwork
from Model.KAN.KANNetwork import KANNetwork


# ─────────────────────────────────────────────
#  1. MODÈLES À COMPARER
# ─────────────────────────────────────────────

MODELS = [
    # ── Baseline MLP ──────────────────────────────────────────────
    {
        "name": "MLP [8-32-32-8]",
        "model_factory": lambda: nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 8),
        ),
        "model_class": nn.Sequential,
        "model_kwargs": {},
    },

    # ── RecurentAttentionNetwork (RAN) ────────────────────────────
    {
        "name": "RAN_small [8-16-8]",
        "model_class": RecurentAttentionNetwork,
        "model_kwargs": {
            "layers_dim": [8, 16, 8],
            "use_layernorm": True,
            "dropout": 0.0,
        },
    },
    {
        "name": "RAN_medium [8-32-32-8]",
        "model_class": RecurentAttentionNetwork,
        "model_kwargs": {
            "layers_dim": [8, 32, 32, 8],
            "use_layernorm": True,
            "dropout": 0.1,
        },
    },
    {
        "name": "RAN_large [8-64-64-64-8]",
        "model_class": RecurentAttentionNetwork,
        "model_kwargs": {
            "layers_dim": [8, 64, 64, 64, 8],
            "use_layernorm": True,
            "dropout": 0.2,
        },
    },

    # ── KAN (Kolmogorov-Arnold Network) ───────────────────────────
    # n_grid=5, k=3 : configuration originale du papier (légère, précise)
    {
        "name": "KAN_small [8-16-8] g5",
        "model_class": KANNetwork,
        "model_kwargs": {
            "layers_dim": [8, 16, 8],
            "n_grid": 5,
            "k": 3,
            "use_layernorm": True,
            "dropout": 0.0,
        },
    },
    {
        "name": "KAN_medium [8-32-32-8] g5",
        "model_class": KANNetwork,
        "model_kwargs": {
            "layers_dim": [8, 32, 32, 8],
            "n_grid": 5,
            "k": 3,
            "use_layernorm": True,
            "dropout": 0.1,
        },
    },
    {
        "name": "KAN_medium [8-32-32-8] g10",
        "model_class": KANNetwork,
        "model_kwargs": {
            "layers_dim": [8, 32, 32, 8],
            "n_grid": 10,       # grille plus fine → plus de capacité
            "k": 3,
            "use_layernorm": True,
            "dropout": 0.1,
        },
    },
    {
        "name": "KAN_large [8-64-64-64-8] g5",
        "model_class": KANNetwork,
        "model_kwargs": {
            "layers_dim": [8, 64, 64, 64, 8],
            "n_grid": 5,
            "k": 3,
            "use_layernorm": True,
            "dropout": 0.2,
        },
    },
]


# ─────────────────────────────────────────────
#  2. TÂCHES SYNTHÉTIQUES
# ─────────────────────────────────────────────

def _make_physics_harmonic(n):
    """Oscillateur harmonique y = sin(w*x) — facile"""
    X    = torch.randn(n, 8)
    freq = torch.tensor([0.5, 1.0, 2.0, 0.3, 1.5, 0.8, 1.2, 0.4])
    y    = torch.sin(X * freq)
    return X, y

def _make_physics_chaotic(n):
    """Interaction non-linéaire croisée — difficile"""
    X = torch.randn(n, 8)
    y = (
        torch.sin(X[:, :4] * X[:, 4:])
        + 0.3 * (X[:, :4] ** 2)
        - 0.5 * torch.cos(X[:, 4:] * 3.14)
        + 0.1 * torch.randn(n, 4)
    )
    return X, y

def _make_polynomial_easy(n):
    """Régression polynomiale deg-2 — facile"""
    X = torch.randn(n, 8)
    W = torch.randn(8, 8) * 0.3
    y = (X @ W) + 0.5 * (X ** 2)
    return X, y

def _make_polynomial_hard(n):
    """Régression polynomiale deg-4 avec interactions — difficile"""
    X  = torch.randn(n, 8)
    W2 = torch.randn(8, 8) * 0.2
    W4 = torch.randn(8, 8) * 0.05
    y  = (X @ W2) + ((X ** 2) @ W2) + ((X ** 4) @ W4)
    return X, y

def _make_spectral_decomp(n):
    """Décomposition fréquentielle — moyen"""
    t     = torch.linspace(0, 2 * 3.14159, 8).unsqueeze(0).expand(n, -1)
    freqs = torch.randint(1, 5, (n, 1)).float()
    X     = torch.sin(freqs * t) + 0.2 * torch.randn(n, 8)
    y     = torch.cos(freqs * t)
    return X, y

def _make_lorenz_attractor(n):
    """Attracteur de Lorenz — très difficile"""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt    = 0.01
    state = torch.randn(n, 3) * 5
    X_list = []
    for _ in range(8):
        dx = sigma * (state[:, 1] - state[:, 0])
        dy = state[:, 0] * (rho - state[:, 2]) - state[:, 1]
        dz = state[:, 0] * state[:, 1] - beta * state[:, 2]
        state = state + dt * torch.stack([dx, dy, dz], dim=1).clamp(-50, 50)
        X_list.append(state)
    X = torch.cat(X_list, dim=1)[:, :8]
    for _ in range(5):
        dx = sigma * (state[:, 1] - state[:, 0])
        dy = state[:, 0] * (rho - state[:, 2]) - state[:, 1]
        dz = state[:, 0] * state[:, 1] - beta * state[:, 2]
        state = state + dt * torch.stack([dx, dy, dz], dim=1).clamp(-50, 50)
    y = state[:, :8] if state.shape[1] >= 8 else torch.cat(
        [state, torch.zeros(n, 8 - state.shape[1])], dim=1)
    return X, y


TASKS = [
    {
        "name": "Harmonic Oscillator",
        "difficulty": "easy",
        "input_dim": 8, "output_dim": 8,
        "generator": _make_physics_harmonic,
        "loss_fn": nn.MSELoss(),
    },
    {
        "name": "Polynomial Regression (deg-2)",
        "difficulty": "easy",
        "input_dim": 8, "output_dim": 8,
        "generator": _make_polynomial_easy,
        "loss_fn": nn.MSELoss(),
    },
    {
        "name": "Spectral Decomposition",
        "difficulty": "medium",
        "input_dim": 8, "output_dim": 8,
        "generator": _make_spectral_decomp,
        "loss_fn": nn.MSELoss(),
    },
    {
        "name": "Polynomial Regression (deg-4)",
        "difficulty": "hard",
        "input_dim": 8, "output_dim": 8,
        "generator": _make_polynomial_hard,
        "loss_fn": nn.MSELoss(),
    },
    {
        "name": "Chaotic Interaction",
        "difficulty": "hard",
        "input_dim": 8, "output_dim": 4,
        "generator": _make_physics_chaotic,
        "loss_fn": nn.MSELoss(),
    },
    {
        "name": "Lorenz Attractor",
        "difficulty": "very_hard",
        "input_dim": 8, "output_dim": 8,
        "generator": _make_lorenz_attractor,
        "loss_fn": nn.MSELoss(),
    },
]


# ─────────────────────────────────────────────
#  3. ENTRAÎNEMENT
# ─────────────────────────────────────────────

TRAIN_CONFIG = {
    "n_samples"   : 10000,
    "epochs"      : 500,
    "batch_size"  : 256,
    "optimizer"   : "adam",
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "scheduler"   : None,
    "patience"    : 40,
    "device"      : "cpu",
    "seed"        : 42,
}


# ─────────────────────────────────────────────
#  4. CROSS-VALIDATION
# ─────────────────────────────────────────────

CROSS_VAL = {
    "n_folds"  : 5,
    "shuffle"  : True,
    "val_ratio": 0.15,
}
