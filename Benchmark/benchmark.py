"""
benchmark.py  —  Pipeline automatisé de recherche pour modèles PyTorch
=======================================================================

Usage
-----
    python benchmark.py experiment_config.py
    python benchmark.py my_custom_config.py --output results/
    python benchmark.py experiment_config.py --no-plots

Ce script charge un fichier de config Python, exécute une cross-validation
complète pour chaque combinaison (modèle × tâche), et génère :
  • Un rapport CSV  : benchmark_results.csv
  • Un rapport JSON : benchmark_results.json
  • Des graphiques  : plots/ (courbes de loss, comparatifs, heatmap)
  • Un résumé texte : benchmark_summary.txt

Format du fichier de config attendu (voir experiment_config.py) :
  - MODELS       : List[dict]
  - TASKS        : List[dict]
  - TRAIN_CONFIG : dict
  - CROSS_VAL    : dict
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_model_depth(model: nn.Module) -> int:
    """Compte le nombre de couches avec paramètres."""
    return sum(1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)))


def load_config(path: str) -> Any:
    """Charge dynamiquement un fichier de config Python."""
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def build_model(model_cfg: dict) -> nn.Module:
    """Instancie un modèle depuis sa config."""
    if "model_factory" in model_cfg and model_cfg["model_factory"] is not None:
        return model_cfg["model_factory"]()
    return model_cfg["model_class"](**model_cfg["model_kwargs"])


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    lr   = cfg.get("lr", 1e-3)
    wd   = cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Optimizer inconnu : {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    name = cfg.get("scheduler", None)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def kfold_indices(n: int, k: int, shuffle: bool, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Génère les indices de K-Fold manuellement."""
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end   = val_start + fold_size if i < k - 1 else n
        val_idx   = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
        folds.append((train_idx, val_idx))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        mse  = nn.functional.mse_loss(y_pred, y_true).item()
        mae  = nn.functional.l1_loss(y_pred, y_true).item()
        rmse = math.sqrt(mse)

        # R² (coefficient de détermination)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

        # Relative error (%)
        rel_err = (torch.abs(y_pred - y_true) / (torch.abs(y_true) + 1e-8)).mean().item() * 100

        # Stabilité numérique
        has_nan = torch.isnan(y_pred).any().item()
        has_inf = torch.isinf(y_pred).any().item()

    return {
        "mse"     : round(mse, 6),
        "rmse"    : round(rmse, 6),
        "mae"     : round(mae, 6),
        "r2"      : round(r2, 4),
        "rel_err%": round(rel_err, 3),
        "nan"     : int(has_nan),
        "inf"     : int(has_inf),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT D'UN FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(
    model       : nn.Module,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    loss_fn     : nn.Module,
    train_cfg   : dict,
    device      : torch.device,
    verbose     : bool = False,
) -> Dict[str, Any]:
    """Entraîne un modèle sur un fold et retourne les métriques et courbes."""

    epochs  = train_cfg.get("epochs", 100)
    patience = train_cfg.get("patience", 0)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs)

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    t_start        = time.time()

    if device != "mps":
        model = torch.compile(model)
        
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue          # skip batch instable
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()
                n_val    += 1
        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        if scheduler:
            scheduler.step()

        # ── Early stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"      Early stop à l'epoch {epoch}/{epochs}")
            break

    # Restaurer le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - t_start

    # Convergence speed : epoch où val_loss < 10% de son max initial
    convergence_epoch = epochs
    if len(val_losses) > 1:
        threshold = val_losses[0] * 0.10
        for ep, vl in enumerate(val_losses):
            if vl <= threshold:
                convergence_epoch = ep + 1
                break

    return {
        "train_losses"    : train_losses,
        "val_losses"      : val_losses,
        "best_val_loss"   : best_val_loss,
        "training_time_s" : round(training_time, 3),
        "epochs_run"      : len(train_losses),
        "convergence_epoch": convergence_epoch,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION FINALE SUR LE TEST DU FOLD
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fold(
    model      : nn.Module,
    test_loader: DataLoader,
    device     : torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            all_pred.append(pred.cpu())
            all_true.append(yb.cpu())
    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)
    return compute_metrics(y_pred, y_true), y_pred, y_true


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(
    model_cfg : dict,
    task      : dict,
    train_cfg : dict,
    cv_cfg    : dict,
    device    : torch.device,
    verbose   : bool = True,
) -> Dict[str, Any]:
    """
    Exécute une K-Fold cross-validation pour un (modèle, tâche) donné.
    Retourne un dict complet de résultats agrégés et par fold.
    """
    set_seed(train_cfg.get("seed", 42))

    n_samples  = train_cfg.get("n_samples", 1000)
    batch_size = train_cfg.get("batch_size", 32)
    n_folds    = cv_cfg.get("n_folds", 5)
    shuffle    = cv_cfg.get("shuffle", True)
    val_ratio  = cv_cfg.get("val_ratio", 0.1)
    seed       = train_cfg.get("seed", 42)

    # Génération du dataset complet
    X, y = task["generator"](n_samples)
    X, y = X.float(), y.float()

    # Ajuster les dims si le modèle diffère de la tâche
    task_in  = task.get("input_dim", X.shape[1])
    task_out = task.get("output_dim", y.shape[1] if y.dim() > 1 else 1)

    # Vérification de compatibilité
    if X.shape[1] != task_in:
        X = X[:, :task_in] if X.shape[1] > task_in else torch.cat([X, torch.zeros(n_samples, task_in - X.shape[1])], 1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    if y.shape[1] != task_out:
        y = y[:, :task_out] if y.shape[1] > task_out else torch.cat([y, torch.zeros(n_samples, task_out - y.shape[1])], 1)

    dataset    = TensorDataset(X, y)
    fold_idxs  = kfold_indices(n_samples, n_folds, shuffle, seed)
    loss_fn    = task["loss_fn"]

    # Infos modèle (calculées une fois)
    _tmp_model = build_model(model_cfg)
    param_info = count_parameters(_tmp_model)
    depth      = get_model_depth(_tmp_model)
    del _tmp_model

    fold_results = []

    for fold_i, (train_val_idx, test_idx) in enumerate(fold_idxs):
        if verbose:
            print(f"    Fold {fold_i+1}/{n_folds} ...", end=" ", flush=True)

        # Split train_val → train / val interne
        n_tv  = len(train_val_idx)
        n_val = max(1, int(n_tv * val_ratio))
        rng   = np.random.default_rng(seed + fold_i)
        rng.shuffle(train_val_idx)
        val_idx_inner   = train_val_idx[:n_val]
        train_idx_inner = train_val_idx[n_val:]

        train_loader = DataLoader(Subset(dataset, train_idx_inner), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx_inner),   batch_size=batch_size)
        test_loader  = DataLoader(Subset(dataset, test_idx),        batch_size=batch_size)

        # Nouveau modèle pour chaque fold (pas de fuite inter-fold)
        model = build_model(model_cfg).to(device)

        # Entraînement
        fold_train = train_fold(model, train_loader, val_loader, loss_fn, train_cfg, device)

        # Évaluation sur le test du fold
        fold_test_metrics, _, _ = evaluate_fold(model, test_loader, device)

        fold_result = {
            "fold"            : fold_i + 1,
            "epochs_run"      : fold_train["epochs_run"],
            "convergence_epoch": fold_train["convergence_epoch"],
            "best_val_loss"   : round(fold_train["best_val_loss"], 6),
            "training_time_s" : fold_train["training_time_s"],
            "train_losses"    : [round(l, 6) for l in fold_train["train_losses"]],
            "val_losses"      : [round(l, 6) for l in fold_train["val_losses"]],
            **{f"test_{k}": v for k, v in fold_test_metrics.items()},
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"MSE={fold_test_metrics['mse']:.4f}  R²={fold_test_metrics['r2']:.4f}  "
                  f"t={fold_train['training_time_s']:.1f}s")

    # ── Agrégation statistique ──
    metrics_keys = [k for k in fold_results[0] if k.startswith("test_") and k not in ("test_nan", "test_inf")]
    aggregated   = {}
    for mk in metrics_keys:
        vals = [fr[mk] for fr in fold_results]
        clean_k = mk.replace("test_", "")
        aggregated[f"{clean_k}_mean"] = round(float(np.mean(vals)), 6)
        aggregated[f"{clean_k}_std"]  = round(float(np.std(vals)),  6)
        aggregated[f"{clean_k}_min"]  = round(float(np.min(vals)),  6)
        aggregated[f"{clean_k}_max"]  = round(float(np.max(vals)),  6)

    aggregated["training_time_mean_s"]   = round(np.mean([fr["training_time_s"] for fr in fold_results]), 3)
    aggregated["convergence_epoch_mean"] = round(np.mean([fr["convergence_epoch"] for fr in fold_results]), 1)
    aggregated["epochs_run_mean"]        = round(np.mean([fr["epochs_run"] for fr in fold_results]), 1)
    aggregated["nan_count"]              = sum(fr["test_nan"] for fr in fold_results)
    aggregated["inf_count"]              = sum(fr["test_inf"] for fr in fold_results)
    aggregated["stability_score"]        = round(1.0 - (aggregated["nan_count"] + aggregated["inf_count"]) / n_folds, 2)

    return {
        "model_name"    : model_cfg["name"],
        "task_name"     : task["name"],
        "task_difficulty": task.get("difficulty", "unknown"),
        "n_params"      : param_info["total"],
        "n_trainable"   : param_info["trainable"],
        "model_depth"   : depth,
        "n_folds"       : n_folds,
        "n_samples"     : n_samples,
        "aggregated"    : aggregated,
        "per_fold"      : fold_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION DES RAPPORTS
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(all_results: List[Dict], out_dir: Path) -> None:
    """Sauvegarde un CSV plat avec une ligne par (modèle × tâche)."""
    rows = []
    for r in all_results:
        agg = r["aggregated"]
        row = {
            "model"          : r["model_name"],
            "task"           : r["task_name"],
            "difficulty"     : r["task_difficulty"],
            "n_params"       : r["n_params"],
            "n_trainable"    : r["n_trainable"],
            "model_depth"    : r["model_depth"],
            "n_folds"        : r["n_folds"],
            **agg,
        }
        rows.append(row)

    csv_path = out_dir / "benchmark_results.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"  → CSV sauvegardé : {csv_path}")


def save_json(all_results: List[Dict], out_dir: Path) -> None:
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  → JSON sauvegardé : {json_path}")


def save_summary(all_results: List[Dict], out_dir: Path, train_cfg: dict, cv_cfg: dict) -> None:
    """Génère un rapport texte lisible."""
    lines = []
    lines.append("=" * 72)
    lines.append("  BENCHMARK REPORT  —  RecurentAttentionNetwork Research Pipeline")
    lines.append("=" * 72)
    lines.append(f"  Samples/tâche : {train_cfg.get('n_samples', '?')}  |  "
                 f"Folds : {cv_cfg.get('n_folds', '?')}  |  "
                 f"Epochs max : {train_cfg.get('epochs', '?')}  |  "
                 f"Optimizer : {train_cfg.get('optimizer', '?')}")
    lines.append("")

    # Grouper par tâche
    tasks_seen = list(dict.fromkeys(r["task_name"] for r in all_results))
    for task_name in tasks_seen:
        task_results = [r for r in all_results if r["task_name"] == task_name]
        diff = task_results[0]["task_difficulty"]
        lines.append(f"┌─ TÂCHE : {task_name}  [{diff}]")
        lines.append(f"│  {'Modèle':<35} {'MSE μ±σ':>20} {'R² μ':>9} {'MAE μ':>9} "
                     f"{'Converg.':>10} {'Params':>8} {'Stable':>7}")
        lines.append("│  " + "─" * 68)
        # Trier par MSE croissant
        task_results_sorted = sorted(task_results, key=lambda r: r["aggregated"].get("mse_mean", float("inf")))
        for i, r in enumerate(task_results_sorted):
            agg     = r["aggregated"]
            rank    = "🥇" if i == 0 else "  "
            mse_str = f"{agg.get('mse_mean', 0):.4f}±{agg.get('mse_std', 0):.4f}"
            r2_str  = f"{agg.get('r2_mean', 0):.3f}"
            mae_str = f"{agg.get('mae_mean', 0):.4f}"
            conv    = f"ep{agg.get('convergence_epoch_mean', 0):.0f}"
            params  = f"{r['n_params']:,}"
            stable  = "✓" if agg.get("stability_score", 1) == 1.0 else f"{agg.get('stability_score', 0):.1f}"
            lines.append(f"│  {rank} {r['model_name']:<33} {mse_str:>20} {r2_str:>9} "
                         f"{mae_str:>9} {conv:>10} {params:>8} {stable:>7}")
        lines.append("│")

    # Récap global par modèle
    lines.append("")
    lines.append("━" * 72)
    lines.append("  RÉCAP GLOBAL (moyenné sur toutes les tâches)")
    lines.append("━" * 72)
    models_seen = list(dict.fromkeys(r["model_name"] for r in all_results))
    global_rows = []
    for model_name in models_seen:
        mresults = [r for r in all_results if r["model_name"] == model_name]
        avg_mse  = np.mean([r["aggregated"].get("mse_mean", 0)  for r in mresults])
        avg_r2   = np.mean([r["aggregated"].get("r2_mean", 0)   for r in mresults])
        avg_time = np.mean([r["aggregated"].get("training_time_mean_s", 0) for r in mresults])
        stab     = np.mean([r["aggregated"].get("stability_score", 1)  for r in mresults])
        params   = mresults[0]["n_params"]
        global_rows.append((model_name, avg_mse, avg_r2, avg_time, stab, params))

    global_rows.sort(key=lambda x: x[1])   # trier par MSE
    lines.append(f"  {'Modèle':<35} {'Avg MSE':>10} {'Avg R²':>9} {'Avg t(s)':>10} "
                 f"{'Stabilité':>10} {'Params':>8}")
    lines.append("  " + "─" * 68)
    for i, (name, mse, r2, t, st, p) in enumerate(global_rows):
        rank = "🥇" if i == 0 else "  "
        lines.append(f"  {rank} {name:<33} {mse:>10.4f} {r2:>9.3f} {t:>10.1f} "
                     f"{st:>10.2f} {p:>8,}")

    lines.append("")
    lines.append("=" * 72)

    summary_path = out_dir / "benchmark_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # Afficher aussi dans la console
    print("\n" + "\n".join(lines))
    print(f"\n  → Résumé sauvegardé : {summary_path}")


def save_plots(all_results: List[Dict], out_dir: Path) -> None:
    """Génère les graphiques matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  ⚠ matplotlib non disponible, plots ignorés.")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Heatmap MSE (modèles × tâches) ──
    models = list(dict.fromkeys(r["model_name"] for r in all_results))
    tasks  = list(dict.fromkeys(r["task_name"]  for r in all_results))

    matrix = np.full((len(models), len(tasks)), np.nan)
    for r in all_results:
        i = models.index(r["model_name"])
        j = tasks.index(r["task_name"])
        matrix[i, j] = r["aggregated"].get("mse_mean", np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.8), max(5, len(models) * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(tasks)));   ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)));  ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Heatmap MSE (modèles × tâches)  — plus bas = meilleur", fontsize=12)
    plt.colorbar(im, ax=ax, label="MSE moyen (K-Fold)")
    for i in range(len(models)):
        for j in range(len(tasks)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > np.nanmedian(matrix) else "black")
    plt.tight_layout()
    fig.savefig(plots_dir / "heatmap_mse.png", dpi=150)
    plt.close(fig)
    print(f"  → Plot : {plots_dir / 'heatmap_mse.png'}")

    # ── 2. Courbes de loss par (modèle × tâche) ──
    for r in all_results:
        model_slug = r["model_name"].replace(" ", "_").replace("/", "-")
        task_slug  = r["task_name"].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        fig, axes  = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{r['model_name']}  ×  {r['task_name']}", fontsize=11)
        colors = cm.tab10(np.linspace(0, 1, r["n_folds"]))

        for fold in r["per_fold"]:
            c = colors[fold["fold"] - 1]
            axes[0].plot(fold["train_losses"], color=c, alpha=0.8, lw=1.2, label=f"Fold {fold['fold']}")
            axes[1].plot(fold["val_losses"],   color=c, alpha=0.8, lw=1.2, label=f"Fold {fold['fold']}")

        for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)"); ax.set_title(title)
            ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = plots_dir / f"loss_{model_slug}_{task_slug}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    print(f"  → {len(all_results)} courbes de loss sauvegardées dans {plots_dir}")

    # ── 3. Bar chart comparatif R² par tâche ──
    n_tasks = len(tasks)
    n_models = len(models)
    x = np.arange(n_tasks)
    width = 0.8 / n_models
    colors = cm.Set2(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2), 6))
    for mi, model_name in enumerate(models):
        r2_vals = []
        for task_name in tasks:
            match = next((r for r in all_results
                          if r["model_name"] == model_name and r["task_name"] == task_name), None)
            r2_vals.append(match["aggregated"].get("r2_mean", 0) if match else 0)
        offset = (mi - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, r2_vals, width=width * 0.9, label=model_name, color=colors[mi], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R² moyen (K-Fold)"); ax.set_title("Comparaison R² par tâche et modèle")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(1, color="green", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "r2_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  → Plot : {plots_dir / 'r2_comparison.png'}")

    # ── 4. Scatter params vs MSE ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in all_results:
        ax.scatter(r["n_params"], r["aggregated"].get("mse_mean", 0),
                   alpha=0.75, s=60,
                   label=f"{r['model_name']} – {r['task_name']}")
    ax.set_xlabel("Nombre de paramètres"); ax.set_ylabel("MSE moyen")
    ax.set_title("Complexité modèle vs Erreur")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    # Pas de légende (trop d'entrées) mais annotations sur les modèles
    model_agg = {}
    for r in all_results:
        n = r["model_name"]
        if n not in model_agg:
            model_agg[n] = {"params": r["n_params"], "mses": []}
        model_agg[n]["mses"].append(r["aggregated"].get("mse_mean", 0))
    for name, info in model_agg.items():
        ax.annotate(name, xy=(info["params"], np.mean(info["mses"])),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(plots_dir / "params_vs_mse.png", dpi=150)
    plt.close(fig)
    print(f"  → Plot : {plots_dir / 'params_vs_mse.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark automatisé de modèles PyTorch avec cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Chemin vers le fichier de configuration Python")
    parser.add_argument("--output", "-o", default="benchmark_output",
                        help="Répertoire de sortie (défaut : benchmark_output/)")
    parser.add_argument("--no-plots", action="store_true", help="Désactiver la génération de graphiques")
    parser.add_argument("--verbose",  "-v", action="store_true", help="Affichage détaillé")
    args = parser.parse_args()

    # ── Chargement config ──
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Fichier de config introuvable : {config_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK PIPELINE")
    print(f"{'='*60}")
    print(f"  Config : {config_path}")

    cfg = load_config(str(config_path))

    MODELS     = getattr(cfg, "MODELS",      [])
    TASKS      = getattr(cfg, "TASKS",       [])
    TRAIN_CFG  = getattr(cfg, "TRAIN_CONFIG", {})
    CV_CFG     = getattr(cfg, "CROSS_VAL",   {"n_folds": 5})

    if not MODELS:
        print("❌ Aucun modèle défini dans MODELS")
        sys.exit(1)
    if not TASKS:
        print("❌ Aucune tâche définie dans TASKS")
        sys.exit(1)

    device = torch.device(TRAIN_CFG.get("device", "cpu"))
    print(f"  Device  : {device}")
    print(f"  Modèles : {len(MODELS)}")
    print(f"  Tâches  : {len(TASKS)}")
    print(f"  Folds   : {CV_CFG.get('n_folds', 5)}")
    print(f"  Total expériences : {len(MODELS) * len(TASKS)}")
    print(f"{'='*60}\n")

    # ── Répertoire de sortie ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Boucle principale ──
    all_results = []
    total_exp   = len(MODELS) * len(TASKS)
    exp_idx     = 0

    t_global_start = time.time()

    for task in TASKS:
        # Filtrer les modèles compatibles avec la tâche (check input/output dim si définis)
        task_in  = task.get("input_dim",  None)
        task_out = task.get("output_dim", None)

        for model_cfg in MODELS:
            exp_idx += 1
            print(f"[{exp_idx:03d}/{total_exp:03d}]  "
                  f"Modèle: {model_cfg['name']:<35}  Tâche: {task['name']}")

            try:
                result = run_cross_validation(
                    model_cfg = model_cfg,
                    task      = task,
                    train_cfg = TRAIN_CFG,
                    cv_cfg    = CV_CFG,
                    device    = device,
                    verbose   = args.verbose,
                )
                all_results.append(result)
                agg = result["aggregated"]
                print(f"         ✓  MSE={agg['mse_mean']:.4f}±{agg['mse_std']:.4f}  "
                      f"R²={agg['r2_mean']:.3f}  "
                      f"t_moy={agg['training_time_mean_s']:.1f}s  "
                      f"stability={agg['stability_score']}")

            except Exception as e:
                print(f"         ✗  ERREUR : {e}")
                if args.verbose:
                    import traceback; traceback.print_exc()

    t_total = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"  Expériences terminées : {len(all_results)}/{total_exp}")
    print(f"  Temps total : {t_total:.1f}s")
    print(f"{'='*60}\n")

    if not all_results:
        print("❌ Aucun résultat à sauvegarder.")
        sys.exit(1)

    # ── Sauvegarde des résultats ──
    print("Sauvegarde des résultats...")
    save_csv(all_results, out_dir)
    save_json(all_results, out_dir)
    save_summary(all_results, out_dir, TRAIN_CFG, CV_CFG)

    if not args.no_plots:
        print("\nGénération des graphiques...")
        save_plots(all_results, out_dir)

    print(f"\n✅ Benchmark terminé. Résultats dans : {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
