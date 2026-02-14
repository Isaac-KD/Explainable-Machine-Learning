"""
train_utils.py — Harness d'entraînement production pour giga-datasets
======================================================================
Fonctionnalités :
  ✓ Détection automatique CPU / CUDA / MPS → chaque optimisation s'active
    uniquement si le matériel la supporte (pas de ralentissement sur CPU)
  ✓ AMP (bfloat16/float16) uniquement sur CUDA/MPS
  ✓ torch.compile uniquement sur CUDA (overhead pur sur CPU)
  ✓ pin_memory uniquement sur CUDA (coûteux sur CPU)
  ✓ AdamW fused uniquement sur CUDA
  ✓ TF32 / cudnn.benchmark uniquement sur CUDA
  ✓ Dataset memory-mapped → charge des TB sans saturer la RAM
  ✓ Gradient accumulation → batch effectif arbitrairement grand
  ✓ DistributedDataParallel (DDP) multi-GPU prêt à l'emploi
  ✓ Gradient clipping + scheduler cosine avec warmup
  ✓ Checkpointing (save/load) compatible DDP

Utilisation recommandée par matériel :

    # CPU (dev / petits datasets)
    config = TrainConfig()   # défauts conservateurs, aucune option GPU

    # GPU (production)
    config = TrainConfig(dtype="bfloat16", compile=True, use_checkpointing=True)
"""

from __future__ import annotations

import os
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Détection du matériel — point central, tout le reste s'y adapte
# ─────────────────────────────────────────────────────────────────────────────

ON_CUDA = torch.cuda.is_available()
ON_MPS  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
ON_CPU  = not ON_CUDA and not ON_MPS


def detect_device() -> torch.device:
    if ON_CUDA: return torch.device("cuda")
    if ON_MPS:  return torch.device("mps")
    return torch.device("cpu")


def _safe_compile(model: nn.Module, mode: str = "max-autotune") -> nn.Module:
    """
    torch.compile uniquement sur CUDA.

    Sur CPU :
      - max-autotune génère des traces Triton inutilisables → overhead pur
      - Même 'default' peut induire une compilation longue sans gain mesurable
        car les goulots d'étranglement CPU sont le cache L1/L2, pas les kernels

    Sur MPS (Apple Silicon) :
      - torch.compile n'est pas encore stable sur MPS → ignoré
    """
    if not ON_CUDA:
        logger.info("torch.compile ignoré (CUDA non disponible)")
        return model
    if not hasattr(torch, "compile"):
        logger.info("torch.compile ignoré (PyTorch < 2.0)")
        return model
    logger.info(f"torch.compile activé (mode={mode})")
    return torch.compile(model, mode=mode, fullgraph=True)  # type: ignore[return-value]


def _safe_dtype(requested: str) -> torch.dtype:
    """
    Retourne le dtype adapté au matériel.

    CPU  → force float32  (float16 est 2-4x plus lent sur CPU sans AVX-512 FP16)
           Exception : bfloat16 est supporté nativement sur Intel Sapphire Rapids
           via AMX, mais reste expérimental — on reste float32 par défaut.
    MPS  → bfloat16 depuis PyTorch 2.3
    CUDA → le dtype demandé
    """
    if ON_CPU:
        if requested != "float32":
            logger.warning(
                f"dtype='{requested}' demandé mais CPU détecté "
                f"-> float32 forcé. "
                f"float16/bfloat16 sont lents sur CPU sans support matériel dédié."
            )
        return torch.float32

    if ON_MPS:
        return torch.bfloat16 if requested == "bfloat16" else torch.float32

    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[requested]


def _configure_backends() -> None:
    """Active les optimisations backend — ignoré silencieusement sur CPU/MPS."""
    if ON_CUDA:
        torch.backends.cudnn.benchmark        = True   # autoselect fastest conv algo
        torch.backends.cuda.matmul.allow_tf32 = True   # +50% GEMM sur Ampere+
        torch.backends.cudnn.allow_tf32       = True
        logger.info("CUDA backends: benchmark=True, TF32=True")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration centralisée
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Chemins
    data_path:         str  = "data/train.npy"
    labels_path:       str  = "data/labels.npy"
    checkpoint_dir:    str  = "checkpoints/"

    # Dimensions
    input_dim:         int  = 128
    layers_dim:        list = field(default_factory=lambda: [128, 512, 256, 64])

    # Entraînement
    epochs:            int   = 50
    batch_size:        int   = 2048
    grad_accum_steps:  int   = 4
    lr:                float = 3e-4
    weight_decay:      float = 1e-5
    max_grad_norm:     float = 1.0
    warmup_steps:      int   = 500
    dropout:           float = 0.1
    use_layernorm:     bool  = True
    use_checkpointing: bool  = False   # GPU: économise la VRAM | CPU: éviter

    # ── Matériel ──────────────────────────────────────────────────────────────
    # Les défauts sont CPU-safe. Pour GPU : dtype="bfloat16", compile=True, pin_memory=True
    dtype:             str  = "float32"  # CPU → auto-forcé float32
    compile:           bool = False       # GPU only
    num_workers:       int  = 4
    prefetch_factor:   int  = 2
    pin_memory:        bool = False       # True uniquement sur CUDA

    def __post_init__(self) -> None:
        """Corrige automatiquement les options incompatibles avec le matériel."""
        if ON_CPU:
            if self.pin_memory:
                logger.warning("pin_memory=True ignoré sur CPU")
                self.pin_memory = False
            if self.compile:
                logger.warning("compile=True ignoré sur CPU")
                self.compile = False
            if self.use_checkpointing:
                logger.warning(
                    "use_checkpointing=True sur CPU ralentit (~+30%). "
                    "Désactivez si la RAM n'est pas contrainte."
                )
        if self.pin_memory and not ON_CUDA:
            self.pin_memory = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset memory-mapped
# ─────────────────────────────────────────────────────────────────────────────

class MemMapDataset(Dataset):
    """
    Dataset basé sur numpy.memmap.
    Le fichier n'est jamais entièrement chargé en mémoire ; le kernel OS
    pagine uniquement les blocs nécessaires → efficace pour des TB de données.
    Compatible CPU et GPU sans modification.

    Création d'un fichier de test :
        import numpy as np
        N, D = 10_000_000, 128
        np.save("data/train.npy",  np.random.randn(N, D).astype(np.float32))
        np.save("data/labels.npy", np.random.randint(0, 10, N).astype(np.int64))
    """

    def __init__(
        self,
        features_path: str,
        labels_path:   str,
        transform:     Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.X = np.load(features_path, mmap_mode="r")
        self.y = np.load(labels_path,   mmap_mode="r")
        self.transform = transform
        assert len(self.X) == len(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = torch.from_numpy(self.X[idx].copy())
        y = torch.tensor(self.y[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    dataset:    Dataset,
    config:     TrainConfig,
    is_train:   bool = True,
    world_size: int  = 1,
    rank:       int  = 0,
) -> DataLoader:
    """
    pin_memory=True uniquement sur CUDA.
    Sur CPU, pin_memory alloue de la mémoire non-paginable qui est coûteuse
    à allouer et inutile (pas de transfert PCIe à optimiser).
    """
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=is_train
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,         # False sur CPU, True sur CUDA
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        drop_last=is_train,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler cosine + warmup
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer:    torch.optim.Optimizer,
    total_steps:  int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(total_steps - warmup_steps, 1),
                               eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Trainer production-grade — CPU et GPU.

    Résumé des comportements par matériel :

    ┌─────────────────────────┬─────────────────────┬────────────────────┐
    │ Optimisation            │ CPU                 │ CUDA               │
    ├─────────────────────────┼─────────────────────┼────────────────────┤
    │ dtype                   │ float32 (forcé)     │ bfloat16 / float16 │
    │ AMP autocast            │ désactivé           │ activé             │
    │ GradScaler              │ désactivé           │ float16 seulement  │
    │ torch.compile           │ ignoré (overhead)   │ max-autotune       │
    │ pin_memory              │ False               │ True               │
    │ AdamW fused             │ False               │ True               │
    │ TF32 / cudnn.benchmark  │ ignoré              │ activé             │
    │ Gradient checkpointing  │ déconseillé         │ recommandé si OOM  │
    └─────────────────────────┴─────────────────────┴────────────────────┘
    """

    def __init__(
        self,
        model:     nn.Module,
        config:    TrainConfig,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        self.config    = config
        self.criterion = criterion or nn.CrossEntropyLoss(label_smoothing=0.1)

        self.device  = detect_device()
        self.rank    = int(os.environ.get("RANK", 0))
        self.world   = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0

        _configure_backends()

        self.amp_dtype = _safe_dtype(config.dtype)
        self.use_amp   = ON_CUDA and self.amp_dtype in (torch.float16, torch.bfloat16)

        if self.is_main:
            hw = "CUDA" if ON_CUDA else ("MPS" if ON_MPS else "CPU")
            logger.info(
                f"Matériel: {hw} | dtype: {self.amp_dtype} | AMP: {self.use_amp}"
            )

        model = model.to(self.device)
        if config.compile:
            model = _safe_compile(model)
        if self.world > 1:
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=False)
        self.model = model

        # AdamW fused uniquement sur CUDA
        use_fused = ON_CUDA and self._supports_fused_adamw()
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (no_decay if ("bias" in name or "norm" in name) else decay).append(p)

        optim_kwargs: dict = dict(lr=config.lr, betas=(0.9, 0.95))
        if use_fused:
            optim_kwargs["fused"] = True

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **optim_kwargs,
        )
        if self.is_main:
            logger.info(f"AdamW fused={use_fused}")

        # GradScaler uniquement pour float16 sur CUDA
        self.scaler = GradScaler(
            enabled=(ON_CUDA and self.amp_dtype == torch.float16)
        )

        self.global_step   = 0
        self.best_val_loss = math.inf

    @staticmethod
    def _supports_fused_adamw() -> bool:
        try:
            torch.optim.AdamW([torch.zeros(1, device="cuda")], lr=1e-3, fused=True)
            return True
        except (TypeError, RuntimeError):
            return False

    def _run_batch(self, x: Tensor, y: Tensor, is_train: bool) -> Tuple[Tensor, float]:
        x = x.to(self.device, non_blocking=ON_CUDA)
        y = y.to(self.device, non_blocking=ON_CUDA)

        ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )
        with ctx:
            logits = self.model(x)
            loss   = self.criterion(logits, y) / self.config.grad_accum_steps

        if is_train:
            self.scaler.scale(loss).backward()

        return logits, loss.item() * self.config.grad_accum_steps

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self, loader: DataLoader, scheduler) -> float:
        self.model.train()
        total_loss, t0 = 0.0, time.perf_counter()

        for step, (x, y) in enumerate(loader):
            _, loss = self._run_batch(x, y, is_train=True)
            total_loss += loss

            if (step + 1) % self.config.grad_accum_steps == 0:
                self._optimizer_step()
                scheduler.step()
                self.global_step += 1

                if self.is_main and self.global_step % 50 == 0:
                    elapsed    = time.perf_counter() - t0
                    throughput = self.config.batch_size * 50 / elapsed
                    logger.info(
                        f"step={self.global_step:>6d} | loss={loss:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"{throughput:,.0f} samples/s"
                    )
                    t0 = time.perf_counter()

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            _, loss = self._run_batch(x, y, is_train=False)
            total_loss += loss
        return total_loss / len(loader)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        total_steps = (len(train_loader) // self.config.grad_accum_steps) * self.config.epochs
        scheduler = build_scheduler(self.optimizer, total_steps, self.config.warmup_steps)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

            train_loss = self._train_epoch(train_loader, scheduler)
            val_info = ""
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                val_info = f" | val_loss={val_loss:.4f}"
                if self.is_main and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    val_info += " <- best"

            if self.is_main:
                logger.info(
                    f"Epoch {epoch:>3d}/{self.config.epochs} | "
                    f"train_loss={train_loss:.4f}{val_info}"
                )
                self.save_checkpoint(f"epoch_{epoch:03d}.pt")

    def save_checkpoint(self, filename: str) -> None:
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )
        torch.save(
            {"model": model_state, "optimizer": self.optimizer.state_dict(),
             "scaler": self.scaler.state_dict(), "step": self.global_step,
             "best_loss": self.best_val_loss},
            Path(self.config.checkpoint_dir) / filename,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        target = self.model.module if isinstance(self.model, DDP) else self.model
        target.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step   = ckpt["step"]
        self.best_val_loss = ckpt["best_loss"]
        logger.info(f"Checkpoint chargé : step={self.global_step}")
