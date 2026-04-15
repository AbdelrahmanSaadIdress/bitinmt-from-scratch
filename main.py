"""
main.py
========
Entry point for training the multilingual NMT system.

Usage (smoke test — runs in ~10 min on CPU or any GPU):
    python main.py +smoke=true

Usage (default — full config from configs/base.yaml):
    python main.py

Usage (override any config key at the CLI via Hydra):
    python main.py Training.max_steps=500

Usage (resume from a checkpoint — full state restore):
    python main.py +resume_from=checkpoints/epoch_002.pt

Usage (warm start — load weights only, reset all training state):
    python main.py +warm_start_from=checkpoints/best_bleu.pt

Checkpoint loading modes
-------------------------
    resume_from
        Full resume: restores model weights, optimizer state, scheduler
        state, global_step, and epoch counter.  Use after a crash or
        pre-emption to continue exactly where training stopped.

    warm_start_from
        Weights-only warm start: loads the model weights from a previous
        checkpoint but resets ALL training state to zero — fresh optimizer,
        fresh Noam LR schedule beginning its warmup again, epoch=0.
        Use this to:
            • Fine-tune on a new language pair after pre-training
            • Re-run training with a different hyperparameter config
            • Transfer weights to a model with a different d_model / n_layers
              (set Training.warm_start_strict=false in that case)

    These two flags are mutually exclusive.  Passing both raises an error.

Execution flow
--------------
    1. Load or train SentencePiece tokenizer
    2. Build train / val / test TranslationDatasets
    3. Wrap each dataset in a token-bucket DataLoader
    4. Construct the Transformer model from config
    5. Hand everything to Trainer and call .train()

Persistent files you will see after running
-------------------------------------------
    data/
        raw/              ← raw .txt files used to train SentencePiece
        spm/
            multinmt.model   ← trained SentencePiece model
            multinmt.vocab

    checkpoints/
        best_bleu.pt      ← best checkpoint by validation BLEU
        epoch_000.pt      ← rolling epoch checkpoints

    outputs/              ← Hydra run logs (one folder per run)
"""

import logging
import os
import random
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data.dataset   import build_dataloader, build_datasets, load_opus_pairs
from data.tokenizer import MultilingualTokenizer
from model.transformer import Transformer
from training.trainer  import Trainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smoke-test config overrides
# ---------------------------------------------------------------------------
SMOKE_OVERRIDES = {
    "Data": {
        "max_examples": 1_000,
        "vocab_size":   4_000,
    },
    "Modelling": {
        "d_model":     128,
        "num_heads":     4,
        "num_layers":    2,
        "d_ff":        256,
        "dropout":     0.1,
    },
    "Training": {
        "batch_size":        512,
        "warmup_steps":      100,
        "max_steps":         300,
        "max_epochs":        999,
        "eval_every_n_epochs": 1,
        "log_every_n_steps":  20,
        "use_amp":          False,
    },
    "Wandb": {
        "project": "multinmt-smoke",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    """Return the best available device string and log hardware info."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU detected: %s  (%.1f GB VRAM)", name, mem)
        return "cuda"
    log.warning("No GPU found — training on CPU (smoke test is fine; full training is slow)")
    return "cpu"


def apply_smoke_overrides(config: dict) -> dict:
    """Deep-merge SMOKE_OVERRIDES into config."""
    for section, values in SMOKE_OVERRIDES.items():
        if section not in config:
            config[section] = {}
        config[section].update(values)
    log.info("Smoke-test overrides applied — run will finish in ~10 min.")
    return config


def collect_raw_text_for_sp(
    config:                 dict,
    raw_dir:                Path,
    max_examples_per_pair:  int = 500_000,
) -> List[str]:
    """
    Download a slice of each OPUS pair, write one .txt file per language to
    `raw_dir` (persistent), and return the file paths for SentencePiece training.

    Parameters
    ----------
    config               : dict   Full config.
    raw_dir              : Path   Where to write the .txt files.
    max_examples_per_pair: int    Lines-per-pair cap.

    Returns
    -------
    list of str   Absolute paths to the written raw-text files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["Data"]

    lang_sentences: dict[str, list] = {}

    for pair_cfg in data_cfg["pairs"]:
        src_lang = pair_cfg["src"]
        tgt_lang = pair_cfg["tgt"]

        log.info("Fetching raw text for SP training: %s–%s ...", src_lang, tgt_lang)
        try:
            raw = load_opus_pairs(
                src_lang, tgt_lang,
                split="train",
                cache_dir=str(raw_dir),
                also_reverse=False,
                max_examples=1000000,
            )
        except Exception as exc:
            log.warning("Could not fetch %s–%s: %s", src_lang, tgt_lang, exc)
            continue

        for src_text, tgt_text, sl, tl in raw:
            lang_sentences.setdefault(sl, []).append(src_text)
            lang_sentences.setdefault(tl, []).append(tgt_text)

    if not lang_sentences:
        raise RuntimeError(
            "No raw text collected for SentencePiece training. "
            "Check your Data.pairs config and internet connection."
        )

    output_files: List[str] = []
    for lang, sentences in sorted(lang_sentences.items()):
        fpath = raw_dir / f"raw_{lang}.txt"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
        log.info("  Wrote %d sentences [%s] → %s", len(sentences), lang, fpath)
        output_files.append(str(fpath))

    return output_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 0.  Setup                                                           #
    # ------------------------------------------------------------------ #
    config = OmegaConf.to_container(cfg, resolve=True)

    if cfg.get("smoke", False):
        config = apply_smoke_overrides(config)

    set_seed(config["Training"]["seed"])
    device = detect_device()

    if device == "cpu":
        config["Training"]["use_amp"] = False

    log.info("\n%s", OmegaConf.to_yaml(cfg))

    try:
        project_root = Path(hydra.utils.get_original_cwd())
    except Exception:
        project_root = Path.cwd()

    raw_dir  = Path(config["Data"]["raw_dir"])
    sp_path  = Path(config["Data"]["sp_model_path"])
    ckpt_dir = Path(config["Training"]["checkpoint_dir"])

    config["Data"]["raw_dir"]             = str(raw_dir)
    config["Data"]["sp_model_path"]       = str(sp_path)
    config["Training"]["checkpoint_dir"]  = str(ckpt_dir)

    log.info("Project root  : %s", project_root)
    log.info("Raw data dir  : %s", raw_dir)
    log.info("SP model path : %s", sp_path)
    log.info("Checkpoint dir: %s", ckpt_dir)

    # ------------------------------------------------------------------ #
    # 0b.  Validate checkpoint flags (mutual exclusion)                  #
    # ------------------------------------------------------------------ #
    raw_resume     = cfg.get("resume_from",     None)
    raw_warm_start = cfg.get("warm_start_from", None)

    if raw_resume and raw_warm_start:
        raise ValueError(
            "Cannot use +resume_from and +warm_start_from together.\n"
            "  +resume_from     → full resume (weights + optimizer + step)\n"
            "  +warm_start_from → weights only, training state resets to zero\n"
            "Pick one."
        )

    resume_from:     Optional[Path] = None
    warm_start_from: Optional[Path] = None

    if raw_resume:
        resume_from = Path(str(raw_resume))
        if not resume_from.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_from}")
        log.info("Mode: full resume from '%s'", resume_from)

    if raw_warm_start:
        warm_start_from = Path(str(raw_warm_start))
        if not warm_start_from.exists():
            raise FileNotFoundError(f"warm_start_from checkpoint not found: {warm_start_from}")
        log.info(
            "Mode: warm start from '%s' — weights loaded, training state reset to zero.",
            warm_start_from,
        )

    # ------------------------------------------------------------------ #
    # 1.  Tokenizer                                                       #
    # ------------------------------------------------------------------ #
    max_examples: Optional[int] = config["Data"].get("max_examples", None)

    if sp_path.exists():
        log.info("Loading existing SentencePiece model: %s", sp_path)
        tokenizer = MultilingualTokenizer(sp_path)

    else:
        log.info("SentencePiece model not found — training from scratch ...")
        sp_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = collect_raw_text_for_sp(
            config,
            raw_dir=raw_dir,
            max_examples_per_pair=max_examples,
        )

        tokenizer = MultilingualTokenizer.train(
            input_files=raw_files,
            model_prefix=sp_path.with_suffix(""),
            vocab_size=config["Data"]["vocab_size"],
            character_coverage=config["Data"]["character_coverage"],
            model_type=config["Data"]["sp_model_type"],
        )

    log.info(
        "Tokenizer ready  vocab_size=%d  "
        "pad=%d  bos=%d  eos=%d  lang_ids=%s",
        tokenizer.vocab_size,
        tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id,
        tokenizer.lang_ids,
    )

    # ------------------------------------------------------------------ #
    # 2.  Build datasets                                                  #
    # ------------------------------------------------------------------ #
    log.info(
        "Building datasets (max_examples=%s per pair per split) ...",
        max_examples if max_examples else "unlimited",
    )

    train_ds, val_ds, test_ds = build_datasets(config, tokenizer, max_examples)

    log.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty after tokenisation/filtering. "
            "Check that the corpus downloaded correctly, or lower "
            "Data.min_seq_length."
        )

    # ------------------------------------------------------------------ #
    # 3.  DataLoaders                                                     #
    # ------------------------------------------------------------------ #
    num_workers = 0 if device == "cpu" else 4

    train_loader = build_dataloader(
        train_ds, tokenizer,
        max_tokens=config["Training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        seed=config["Training"]["seed"],
    )
    val_loader = build_dataloader(
        val_ds, tokenizer,
        max_tokens=config["Training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        seed=config["Training"]["seed"],
    )

    log.info(
        "DataLoaders ready — train batches: %d  val batches: %d",
        len(train_loader), len(val_loader),
    )

    # ------------------------------------------------------------------ #
    # 4.  Model                                                           #
    # ------------------------------------------------------------------ #
    config["Modelling"]["src_vocab_size"] = tokenizer.vocab_size
    config["Modelling"]["tgt_vocab_size"] = tokenizer.vocab_size

    model = Transformer(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %.2fM trainable parameters", n_params / 1e6)

    # ------------------------------------------------------------------ #
    # 5.  Train                                                           #
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        resume_from=resume_from,
        warm_start_from=warm_start_from,
    )

    trainer.train()
    log.info("Run complete.  Best BLEU: %.2f", trainer.best_bleu)


if __name__ == "__main__":
    main()