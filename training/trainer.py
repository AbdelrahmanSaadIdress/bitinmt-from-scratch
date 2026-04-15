"""
training/trainer.py
====================
Training loop for the multilingual Transformer NMT system.

Covers (in order of execution):
    • Model + optimizer + scheduler construction from config
    • Mixed-precision training via torch.cuda.amp  (beyond-paper speedup)
    • Gradient accumulation for simulating large batches on small GPUs
    • Gradient clipping (max_norm=1.0, paper Section 5.3 footnote)
    • Checkpoint saving: best-BLEU model + rolling last-N checkpoints
    • Wandb logging: loss, LR, BLEU per language pair, attention heatmaps
    • Per-epoch validation with BLEU evaluation via sacrebleu

Two checkpoint-loading modes
-----------------------------
    resume_from    (Path | None)
        Full resume: restores model weights, optimizer state, scheduler
        state, global_step, epoch, and best_bleu.  Training continues
        exactly where it left off — the LR curve is unbroken.
        Use this after a crash or pre-emption.

    warm_start_from (Path | None)
        Weights-only warm start: loads model weights from a previous run
        but resets *all* training state (step=0, epoch=0, fresh optimizer,
        fresh Noam schedule, best_bleu=0).  Training restarts from scratch
        with the pre-trained weights as the initial point.
        Use this to:
            • Fine-tune on a new language pair
            • Resume with a different learning-rate schedule
            • Experiment with a different batch size / optimizer config
            • Add new language tags and continue from a strong baseline

Paper references:
    Optimizer:  Section 5.3 — Adam β₁=0.9, β₂=0.98, ε=1e-9
    Regularisation: Section 5.4 — dropout=0.1, label smoothing ε=0.1
    Gradient clipping: not in the paper but universal practice; harmless
    Mixed precision: beyond the paper — purely a speed optimisation
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.tokenizer import MultilingualTokenizer
from evaluation.bleu import compute_corpus_bleu
from evaluation.beam_search import greedy_decode
from model.transformer import Transformer
from training.losses import LabelSmoothingLoss
from training.scheduler import build_noam_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:         Path,
    model:        Transformer,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LambdaLR,
    scaler:       Optional[GradScaler],
    global_step:  int,
    epoch:        int,
    best_bleu:    float,
    config:       dict,
) -> None:
    """
    Save all training state to a `.pt` file for resuming or inference.

    Parameters
    ----------
    path        : Path   Where to write the checkpoint.
    model       : Transformer
    optimizer   : Adam optimizer
    scheduler   : Noam LR scheduler
    scaler      : AMP GradScaler (or None if not using AMP)
    global_step : int   Total steps taken so far.
    epoch       : int   Last completed epoch (0-indexed).
    best_bleu   : float Best validation BLEU seen so far.
    config      : dict  Full Hydra config (for reproducibility).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler else None,
        "global_step":     global_step,
        "epoch":           epoch,
        "best_bleu":       best_bleu,
        "config":          config,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved → %s  (step=%d, epoch=%d)", path, global_step, epoch)


def load_checkpoint(
    path:      Path,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler:    Optional[GradScaler] = None,
    device:    str = "cpu",
) -> Tuple[int, int, float]:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).

    Parameters
    ----------
    path      : Path   Checkpoint file.
    model     : Transformer   Will receive the saved weights.
    optimizer : optional      If provided, optimizer state is restored.
    scheduler : optional      If provided, scheduler state is restored.
    scaler    : optional      If provided, AMP scaler state is restored.
    device    : str           Where to map the tensors ('cpu', 'cuda', etc.)

    Returns
    -------
    (global_step, epoch, best_bleu)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    logger.info(
        "Loaded checkpoint '%s'  step=%d  epoch=%d  best_bleu=%.2f",
        path, ckpt["global_step"], ckpt["epoch"], ckpt["best_bleu"],
    )
    return ckpt["global_step"], ckpt["epoch"], ckpt["best_bleu"]


def load_weights_only(
    path:   Path,
    model:  Transformer,
    device: str = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load *only* the model weights from a checkpoint file, discarding all
    training state (optimizer, scheduler, step counter, epoch, best_bleu).

    This is the correct function to call when you want a **warm start**:
    you inherit the learned representations but begin a completely fresh
    training run with a new LR schedule, optimizer state, and epoch counter.

    Why not just call load_checkpoint with optimizer=None?
        load_checkpoint still returns (global_step, epoch, best_bleu) which
        would mislead the trainer into thinking it is mid-run.  This function
        makes the intent explicit and returns the source checkpoint's config
        for reference / sanity-checking only — nothing is applied.

    Parameters
    ----------
    path   : Path         Path to a `.pt` checkpoint file.
    model  : Transformer  Will receive the saved weights in-place.
    device : str          Device to map tensors to while loading.
    strict : bool         If True (default), the state-dict keys must match
                          exactly.  Set False if you have added new layers
                          (e.g. a new language embedding) and want to load
                          only the matching subset — missing keys will be
                          randomly initialised.

    Returns
    -------
    dict   The config that was stored inside the checkpoint, so the caller
           can log it or assert that vocab sizes are compatible.

    Raises
    ------
    FileNotFoundError  if `path` does not exist.
    KeyError           if the file has no 'model_state' key (not a valid
                       checkpoint from this project).
    RuntimeError       if strict=True and the state-dict keys don't match
                       (forwarded from PyTorch's load_state_dict).
    """
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    if "model_state" not in ckpt:
        raise KeyError(
            f"'{path}' does not contain a 'model_state' key. "
            "Is this a valid checkpoint from this project?"
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict)

    if missing:
        logger.warning(
            "Warm start — %d keys NOT found in checkpoint (will use random init): %s",
            len(missing), missing,
        )
    if unexpected:
        logger.warning(
            "Warm start — %d keys in checkpoint NOT in current model (ignored): %s",
            len(unexpected), unexpected,
        )

    src_config = ckpt.get("config", {})
    src_step   = ckpt.get("global_step", "?")
    src_epoch  = ckpt.get("epoch", "?")
    src_bleu   = ckpt.get("best_bleu", 0.0)

    logger.info(
        "Warm start: weights loaded from '%s'  "
        "(original run: step=%s  epoch=%s  best_bleu=%.2f).  "
        "All training state reset to zero.",
        path, src_step, src_epoch, src_bleu,
    )

    return src_config


def _prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    """
    Delete oldest epoch checkpoints, keeping only the `keep` most recent.
    Best-BLEU checkpoint is never deleted (its filename starts with 'best').
    """
    epoch_ckpts = sorted(
        [f for f in ckpt_dir.glob("epoch_*.pt")],
        key=lambda f: f.stat().st_mtime,
    )
    for old in epoch_ckpts[:-keep]:
        old.unlink()
        logger.debug("Pruned old checkpoint: %s", old)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the full training + validation loop.

    Parameters
    ----------
    config          : dict                  Full Hydra config.
    model           : Transformer
    tokenizer       : MultilingualTokenizer
    train_loader    : DataLoader            Token-bucket batched train set.
    val_loader      : DataLoader            Validation set.
    device          : str                   'cuda', 'mps', or 'cpu'.
    resume_from     : Path | None           Full resume (weights + optimizer
                                            + step counter).  Use after crash.
    warm_start_from : Path | None           Weights-only warm start.  Training
                                            state resets to zero.  Mutually
                                            exclusive with resume_from.
    """

    def __init__(
        self,
        config:           dict,
        model:            Transformer,
        tokenizer:        MultilingualTokenizer,
        train_loader:     DataLoader,
        val_loader:       DataLoader,
        device:           str = "cuda",
        resume_from:      Optional[Path] = None,
        warm_start_from:  Optional[Path] = None,
    ) -> None:

        # Guard: both flags set at once is almost certainly a mistake
        if resume_from is not None and warm_start_from is not None:
            raise ValueError(
                "resume_from and warm_start_from are mutually exclusive. "
                "Use resume_from to continue a run after a crash, or "
                "warm_start_from to start fresh from pre-trained weights — "
                "never both at the same time."
            )

        self.config     = config
        self.model      = model.to(device)
        self.tokenizer  = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device     = device

        train_cfg   = config["Training"]
        model_cfg   = config["Modelling"]

        # --- Optimizer (paper Section 5.3) ---
        # Base lr=1.0: the Noam schedule carries all the scaling
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
            eps=train_cfg["adam_eps"],
        )

        # --- LR scheduler (paper eq. 3) ---
        self.scheduler = build_noam_scheduler(
            self.optimizer,
            d_model=model_cfg["d_model"],
            warmup_steps=train_cfg["warmup_steps"],
        )

        # --- Loss (paper Section 5.4) ---
        self.criterion = LabelSmoothingLoss(
            vocab_size=model_cfg["tgt_vocab_size"],
            pad_idx=tokenizer.pad_id,
            smoothing=train_cfg["label_smoothing"],
        )

        # --- Mixed precision (beyond-paper) ---
        self.use_amp   = train_cfg["use_amp"] and device == "cuda"
        self.scaler    = GradScaler("cuda") if self.use_amp else None

        # --- Training state (may be overridden by resume_from below) ---
        self.global_step  = 0
        self.start_epoch  = 0
        self.best_bleu    = 0.0
        self.grad_accum   = train_cfg["grad_accum_steps"]
        self.max_grad_norm = train_cfg["gradient_clip"]

        # --- Config shortcuts ---
        self.max_epochs          = train_cfg["max_epochs"]
        self.max_steps           = train_cfg["max_steps"]
        self.log_every           = train_cfg["log_every_n_steps"]
        self.eval_every_epoch    = train_cfg["eval_every_n_epochs"]
        self.save_every_epoch    = train_cfg["save_every_n_epochs"]
        self.keep_n_ckpts        = train_cfg["keep_last_n_checkpoints"]
        self.ckpt_dir            = Path(train_cfg["checkpoint_dir"])

        # --- Wandb (optional) ---
        self._init_wandb()

        # ------------------------------------------------------------------
        # Checkpoint loading — mutually exclusive branches
        # ------------------------------------------------------------------

        if resume_from is not None:
            # Full resume: weights + optimizer + scheduler + counters.
            # The training curve is unbroken — useful after crashes.
            self.global_step, self.start_epoch, self.best_bleu = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                device,
            )
            # Advance start_epoch by 1: we resume AFTER the saved epoch
            self.start_epoch += 1
            logger.info(
                "Full resume from '%s' — continuing at epoch=%d  step=%d",
                resume_from, self.start_epoch, self.global_step,
            )

        elif warm_start_from is not None:
            # Warm start: weights only — all training state stays at zero.
            # The Noam LR schedule begins its warmup from step 0, the optimizer
            # has no momentum memory, and epoch counting starts from 0.
            src_config = load_weights_only(
                warm_start_from,
                self.model,
                device=device,
                strict=self.config["Training"].get("warm_start_strict", True),
            )
            # Sanity-check: warn if the source model had a different vocab size,
            # since mismatched embedding tables mean the weights for the
            # projection layer are shape-incompatible (load_state_dict would
            # have already raised if strict=True, but a clear log is helpful).
            src_vocab = src_config.get("Modelling", {}).get("tgt_vocab_size")
            cur_vocab = model_cfg["tgt_vocab_size"]
            if src_vocab and src_vocab != cur_vocab:
                logger.warning(
                    "Warm start vocab mismatch: source checkpoint has "
                    "vocab_size=%d but current config has %d.  "
                    "If strict=True this would have raised already.  "
                    "If strict=False, the embedding/projection layers "
                    "are randomly initialised for the new vocab positions.",
                    src_vocab, cur_vocab,
                )
            logger.info(
                "Warm start complete — training state is fresh "
                "(epoch=0, step=0, best_bleu=0.0).",
            )
            # self.global_step / start_epoch / best_bleu remain 0 — by design

        logger.info(
            "Trainer ready | device=%s  amp=%s  grad_accum=%d  "
            "start_epoch=%d  start_step=%d  max_epochs=%d",
            device, self.use_amp, self.grad_accum,
            self.start_epoch, self.global_step, self.max_epochs,
        )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialise wandb if a project name is configured."""
        wandb_cfg = self.config.get("Wandb", {})
        project   = wandb_cfg.get("project", "")
        self._wandb = None

        if not project:
            logger.info("Wandb project not set — skipping wandb logging.")
            return

        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project,
                entity=wandb_cfg.get("entity") or None,
                config=self.config,
                resume="allow",
            )
            # Log model parameter count
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.run.summary["n_params"] = n_params
            logger.info("Wandb initialised: project=%s", project)
        except ImportError:
            logger.warning("wandb not installed — install with: pip install wandb")

    def _wandb_log(self, payload: dict) -> None:
        """Log a dict to wandb if available, otherwise no-op."""
        if self._wandb:
            self._wandb.log(payload, step=self.global_step)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _check_nan(self, name: str, tensor: torch.Tensor) -> bool:
        """
        Return True and log a detailed report if `tensor` contains NaN or Inf.
        Call this at each stage of the forward pass to pinpoint where NaN first
        appears — the earliest positive hit is the root cause.
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            logger.error(
                "NaN/Inf in %-30s | shape=%-20s | "
                "min=%.4g  max=%.4g  mean=%.4g  "
                "nan_count=%d  inf_count=%d",
                name, str(tuple(tensor.shape)),
                tensor[torch.isfinite(tensor)].min().item() if torch.isfinite(tensor).any() else float("nan"),
                tensor[torch.isfinite(tensor)].max().item() if torch.isfinite(tensor).any() else float("nan"),
                tensor[torch.isfinite(tensor)].mean().item() if torch.isfinite(tensor).any() else float("nan"),
                torch.isnan(tensor).sum().item(),
                torch.isinf(tensor).sum().item(),
            )
            return True
        return False

    def _train_step(self, batch: Dict) -> float:
        """
        Forward + backward for one batch.

        Returns
        -------
        float   Loss value (for logging; detached from graph).
        """
        src     = batch["src"].to(self.device)
        tgt_in  = batch["tgt_in"].to(self.device)
        tgt_out = batch["tgt_out"].to(self.device)

        vocab_size = self.config["Modelling"]["src_vocab_size"]
        if (src >= vocab_size).any() or (src < 0).any():
            logger.error(
                "Out-of-range token id in src: min=%d  max=%d  vocab_size=%d",
                src.min().item(), src.max().item(), vocab_size,
            )
        if (tgt_in >= vocab_size).any() or (tgt_in < 0).any():
            logger.error(
                "Out-of-range token id in tgt_in: min=%d  max=%d  vocab_size=%d",
                tgt_in.min().item(), tgt_in.max().item(), vocab_size,
            )

        n_real_tgt_tokens = (tgt_out != self.tokenizer.pad_id).sum().item()
        if n_real_tgt_tokens == 0:
            logger.error(
                "All-padding target batch at step %d — "
                "this produces NaN loss.  "
                "src shape=%s  tgt_out shape=%s",
                self.global_step, tuple(src.shape), tuple(tgt_out.shape),
            )

        src_mask = Transformer.make_src_mask(src, self.tokenizer.pad_id)
        tgt_mask = Transformer.make_tgt_mask(tgt_in, self.tokenizer.pad_id)

        src_emb = self.model.positional_encoding(
            self.model.embedded_enc(src) * math.sqrt(self.model.d_model)
        )
        tgt_emb = self.model.positional_encoding(
            self.model.embedded_dec(tgt_in) * math.sqrt(self.model.d_model)
        )
        nan_in_emb = (
            self._check_nan("src_embedding", src_emb) |
            self._check_nan("tgt_embedding", tgt_emb)
        )

        enc_out = src_emb
        nan_in_enc = False
        for i, enc_layer in enumerate(self.model.encoder_layers):
            enc_out = enc_layer(enc_out, src_mask)
            if self._check_nan(f"encoder_layer_{i}_output", enc_out):
                nan_in_enc = True
                break

        dec_out = tgt_emb
        nan_in_dec = False
        for i, dec_layer in enumerate(self.model.decoder_layers):
            dec_out = dec_layer(dec_out, enc_out, src_mask, tgt_mask)
            if self._check_nan(f"decoder_layer_{i}_output", dec_out):
                nan_in_dec = True
                break

        logits = self.model.out(dec_out)
        self._check_nan("logits", logits)

        raw_loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            tgt_out.view(-1),
        )
        self._check_nan("loss", raw_loss.unsqueeze(0))

        if any([nan_in_emb, nan_in_enc, nan_in_dec,
                not math.isfinite(raw_loss.item())]):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pnorm = param.data.norm().item()
                    gnorm = param.grad.norm().item() if param.grad is not None else 0.0
                    if not math.isfinite(pnorm) or not math.isfinite(gnorm) or pnorm > 1e4:
                        logger.error(
                            "  BAD PARAM  %-50s  |w|=%.3g  |g|=%.3g",
                            name, pnorm, gnorm,
                        )

        scaled_loss = raw_loss / self.grad_accum
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return raw_loss.item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Run greedy-decode on the validation set and compute BLEU per language pair.

        Returns
        -------
        dict mapping "bleu/{src}-{tgt}" → float
        """
        MAX_VAL_BATCHES = 500

        self.model.eval()
        hyp_by_pair: Dict[str, list] = {}
        ref_by_pair: Dict[str, list] = {}

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= MAX_VAL_BATCHES:
                break
            if batch_idx % 50 == 0:
                logger.info("  Validating... batch %d/%d", batch_idx, MAX_VAL_BATCHES)

            src       = batch["src"].to(self.device)
            src_langs = batch["src_langs"]
            tgt_langs = batch["tgt_langs"]
            ref_texts = batch["tgt_texts"]

            src_mask = Transformer.make_src_mask(src, self.tokenizer.pad_id)

            src_len     = src.size(1)
            max_dec_len = min(
                src_len + 50,
                self.config["Evaluation"]["max_decode_steps"],
            )

            pred_ids = greedy_decode(
                model=self.model,
                src=src,
                src_mask=src_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_len=max_dec_len,
                device=self.device,
            )

            for i, pred in enumerate(pred_ids):
                pair_key = f"{src_langs[i]}-{tgt_langs[i]}"
                hyp = self.tokenizer.decode(pred, skip_special_tokens=True)
                hyp = hyp.strip()
                ref = ref_texts[i]
                hyp_by_pair.setdefault(pair_key, []).append(hyp)
                ref_by_pair.setdefault(pair_key, []).append([ref])

        bleu_scores: Dict[str, float] = {}
        for pair_key in hyp_by_pair:
            bleu = compute_corpus_bleu(hyp_by_pair[pair_key], ref_by_pair[pair_key], pair_key)
            bleu_scores[f"bleu/{pair_key}"] = bleu
            logger.info("  [%s]  BLEU = %.2f", pair_key, bleu)

        self.model.train()
        return bleu_scores

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the full training loop from `start_epoch` to `max_epochs`.

        Loop structure:
            for epoch in range(max_epochs):
                for batch in train_loader:
                    forward → loss → backward
                    every grad_accum steps: clip + optimizer.step + scheduler.step
                    every log_every steps:  log loss + LR to wandb
                if epoch % eval_every: validate → log BLEU → maybe save best
                if epoch % save_every: save epoch checkpoint
        """
        logger.info("=== Training start  epochs=%d  steps=%d ===",
                    self.max_epochs, self.max_steps)

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_loss   = 0.0
            epoch_steps  = 0
            epoch_tokens = 0
            t0 = time.time()

            for step_in_epoch, batch in enumerate(self.train_loader):

                loss = self._train_step(batch)

                if math.isfinite(loss):
                    epoch_loss += loss
                else:
                    logger.warning(
                        "Non-finite loss (%.4g) at epoch=%d micro_step=%d — "
                        "skipping accumulation.",
                        loss, epoch, step_in_epoch,
                    )

                epoch_tokens += batch["src"].numel() + batch["tgt_in"].numel()

                micro_step = (step_in_epoch + 1)
                if micro_step % self.grad_accum == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    epoch_steps      += 1

                    if self.global_step % self.log_every == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            "epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                            epoch, self.global_step, loss, current_lr,
                        )
                        self._wandb_log({
                            "train/loss": loss,
                            "train/lr":   current_lr,
                            "train/epoch": epoch,
                        })

                    if self.global_step >= self.max_steps:
                        logger.info("Reached max_steps=%d — stopping.", self.max_steps)
                        self._end_of_epoch(epoch, epoch_loss, epoch_steps, epoch_tokens, t0)
                        return

            self._end_of_epoch(epoch, epoch_loss, epoch_steps, epoch_tokens, t0)

        logger.info("=== Training complete ===")
        if self._wandb:
            self._wandb.finish()

    def _end_of_epoch(
        self,
        epoch:        int,
        epoch_loss:   float,
        epoch_steps:  int,
        epoch_tokens: int,
        t0:           float,
    ) -> None:
        """Validation, BLEU logging, and checkpointing at epoch end."""
        elapsed   = time.time() - t0
        avg_loss  = epoch_loss / max(len(self.train_loader), 1)
        tok_per_s = epoch_tokens / max(elapsed, 1e-6)

        logger.info(
            "── Epoch %d done  avg_loss=%.4f  tokens/s=%.0f  time=%.1fs",
            epoch, avg_loss, tok_per_s, elapsed,
        )

        if (epoch + 1) % self.eval_every_epoch == 0:
            bleu_scores = self._validate(epoch)
            mean_bleu   = sum(bleu_scores.values()) / max(len(bleu_scores), 1)

            self._wandb_log({
                "val/avg_bleu": mean_bleu,
                "val/epoch": epoch,
                **bleu_scores,
            })

            if mean_bleu > self.best_bleu:
                self.best_bleu = mean_bleu
                save_checkpoint(
                    path=self.ckpt_dir / "best_bleu.pt",
                    model=self.model, optimizer=self.optimizer,
                    scheduler=self.scheduler, scaler=self.scaler,
                    global_step=self.global_step, epoch=epoch,
                    best_bleu=self.best_bleu, config=self.config,
                )
                logger.info("  ★ New best BLEU: %.2f", self.best_bleu)

        if (epoch + 1) % self.save_every_epoch == 0:
            save_checkpoint(
                path=self.ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=self.model, optimizer=self.optimizer,
                scheduler=self.scheduler, scaler=self.scaler,
                global_step=self.global_step, epoch=epoch,
                best_bleu=self.best_bleu, config=self.config,
            )
            _prune_checkpoints(self.ckpt_dir, self.keep_n_ckpts)