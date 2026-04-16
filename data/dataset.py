"""
data/dataset.py
================
OPUS corpus loader, PyTorch Dataset, and dynamic-batching DataLoader.

Design decisions:
    1. **Dynamic (token-bucket) batching** — rather than batching by sentence
       count, we group sentence pairs so that the total number of tokens per
       batch stays close to `max_tokens`.  This is how the paper trains
       ("~25,000 source tokens per batch", Section 5.1) and is crucial for
       GPU utilisation: a batch of short sentences and a batch of long ones
       should take similar wall-clock time, not the same sentence count.

    2. **Language-pair interleaving** — all (src_lang, tgt_lang) pairs share
       one DataLoader.  Each example carries explicit lang codes; the
       tokenizer prepends the correct source-language tag automatically.
       This means the model sees en→ar, en→fr, ar→en, fr→en examples
       mixed within a single epoch.

    3. **Streaming-friendly design** — the dataset can work with pre-cached
       tokenised numpy arrays (fast) or raw text files (flexible).  We default
       to raw-text loading so the project has no binary format dependency.

    4. **Reproducible splits** — train/val/test are split by a seeded shuffle
       so results are reproducible across machines.

Note on data sources:
    We use the Helsinki-NLP/opus-100 dataset from Hugging Face Datasets.
    This provides aligned sentence pairs for 100 language pairs; we use
    en-ar and en-fr subsets.  For reverse directions (ar→en, fr→en) we
    simply flip src/tgt at collation time — no extra download needed.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from data.tokenizer import MultilingualTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranslationPair:
    """
    A single tokenised translation example ready for model input.

    Fields
    ------
    src_ids : list of int   Source token IDs (with language tag + EOS).
    tgt_ids : list of int   Target token IDs (with BOS + EOS — teacher forced).
    src_lang : str          Source language code, e.g. 'en'.
    tgt_lang : str          Target language code, e.g. 'ar'.
    src_text : str          Original source string (kept for debugging / BLEU).
    tgt_text : str          Original target string.
    """
    src_ids:  List[int]
    tgt_ids:  List[int]
    src_lang: str
    tgt_lang: str
    src_text: str = ""
    tgt_text: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    PyTorch Dataset over a list of `TranslationPair` objects.

    Typical construction flow:
        raw_pairs = load_opus_pairs(...)
        tokenised  = tokenise_pairs(raw_pairs, tokenizer, max_length=150)
        dataset    = TranslationDataset(tokenised)

    Parameters
    ----------
    pairs : list of TranslationPair
        Pre-tokenised examples.
    """

    def __init__(self, pairs: List[TranslationPair]) -> None:
        self.pairs = pairs
        logger.info("TranslationDataset: %d examples", len(pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> TranslationPair:
        return self.pairs[idx]


# ---------------------------------------------------------------------------
# OPUS data loading helpers
# ---------------------------------------------------------------------------

def load_opus_pairs(
    src_lang: str,
    tgt_lang: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    also_reverse: bool = True,
    ratios: Optional[List[float]] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Download (or load from cache) an OPUS-100 language pair from
    Hugging Face Datasets and return raw sentence pairs.

    Parameters
    ----------
    src_lang : str
        Source language code, e.g. 'en'.
    tgt_lang : str
        Target language code, e.g. 'ar'.
    split : str
        HuggingFace dataset split: 'train', 'validation', or 'test'.
    cache_dir : str | None
        Local cache directory for Hugging Face Datasets.
    max_examples : int | None
        Cap on number of examples (useful for quick experiments / CI).
    also_reverse : bool
        If True, also add the reversed pair (tgt→src) to the returned list.
        This doubles the data and gives the model both translation directions
        "for free".

    Returns
    -------
    list of (src_text, tgt_text, src_lang, tgt_lang)
    """
    try:
        from datasets import load_dataset  # HuggingFace datasets
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets"
        ) from exc
    # OPUS-100 config name format: "{lang1}-{lang2}" (alphabetical order)
    lang_pair = "-".join(sorted([src_lang, tgt_lang]))
    logger.info("Loading OPUS-100 [%s] split=%s ...", lang_pair, split)

    ds = load_dataset(
        "Helsinki-NLP/opus-100",
        lang_pair,
        split="train",
        cache_dir=cache_dir,
    )
    
    if max_examples is not None and ratios is None:
        ds = ds.select(range(min(max_examples, len(ds))))

    if ratios is not None:
        if len(ratios) != 3:
            raise ValueError("ratios must be [train_ratio, val_ratio, test_ratio]")
        ds = ds.shuffle(seed=42)  # Shuffle before splitting to ensure randomness
        train_end = int(ratios[0] * max_examples)                          # 700
        val_end   = train_end + int(ratios[1] * max_examples)              # 700 + 300 = 1000 (but ratios[1] should be 0.2 → 900)
        test_end  = val_end   + int(ratios[2] * max_examples)              # 900 + 100 = 1000

        if split == "train":
            ds = ds.select(range(0, train_end))
        elif split == "validation":
            ds = ds.select(range(train_end, val_end))
        elif split == "test":
            ds = ds.select(range(val_end,   test_end))
        else:
            raise ValueError(f"Invalid split: {split}")


    # OPUS stores both sides under ds["translation"][lang_code]
    raw: List[Tuple[str, str, str, str]] = []
    for item in ds:
        s = item["translation"].get(src_lang, "").strip()
        t = item["translation"].get(tgt_lang, "").strip()
        if s and t:
            raw.append((s, t, src_lang, tgt_lang))
            if also_reverse:
                raw.append((t, s, tgt_lang, src_lang))
    logger.info("Loaded %d sentence pairs (incl. reverse=%s)", len(raw), also_reverse)
    return raw

def tokenise_pairs(
    raw_pairs: List[Tuple[str, str, str, str]],
    tokenizer: MultilingualTokenizer,
    max_length: int = 150,
    min_length: int = 3,
    show_progress: bool = True,
) -> List[TranslationPair]:
    """
    Tokenise a list of raw (src_text, tgt_text, src_lang, tgt_lang) tuples
    into `TranslationPair` objects.

    Encoding conventions:
        source: [<lang_tag>] + tokens + [EOS]
            — language tag is the conditioning signal (Johnson et al. 2017)
            — no BOS on source (tag already identifies start + language)
        target (for teacher forcing): [BOS] + tokens + [EOS]
            — model predicts tokens at positions 1..T given 0..T-1

    Pairs where either side exceeds `max_length` or is shorter than
    `min_length` tokens are filtered out.  Filtering at tokenisation time
    (rather than collation time) avoids wasted GPU cycles on over-long
    sequences.

    Parameters
    ----------
    raw_pairs   : list of (src, tgt, src_lang, tgt_lang)
    tokenizer   : MultilingualTokenizer
    max_length  : int   Filter threshold (inclusive).
    min_length  : int   Minimum token count (exclusive of special tokens).
    show_progress : bool   Log a progress message every 100k examples.

    Returns
    -------
    list of TranslationPair
    """
    results: List[TranslationPair] = []
    filtered = 0

    for i, (src_text, tgt_text, src_lang, tgt_lang) in enumerate(raw_pairs):
        if show_progress and i > 0 and i % 100_000 == 0:
            logger.info("  Tokenised %d / %d  (filtered: %d)", i, len(raw_pairs), filtered)

        src_ids = tokenizer.encode(
            src_text, lang=src_lang, add_bos=False, add_eos=True, max_length=max_length
        )
        tgt_ids = tokenizer.encode(
            tgt_text, lang=None, add_bos=True, add_eos=True, max_length=max_length
        )

        # Filter over-long and near-empty sequences
        if (
            len(src_ids) > max_length
            or len(tgt_ids) > max_length
            or len(src_ids) < min_length
            or len(tgt_ids) < min_length
        ):
            filtered += 1
            continue

        results.append(TranslationPair(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_text=src_text,
            tgt_text=tgt_text,
        ))

    logger.info(
        "Tokenisation complete: %d kept, %d filtered (max_len=%d, min_len=%d)",
        len(results), filtered, max_length, min_length,
    )
    return results


# ---------------------------------------------------------------------------
# Dynamic (token-bucket) batching sampler
# ---------------------------------------------------------------------------

class TokenBucketSampler(Sampler):
    """
    Yields batches of example indices such that the total number of tokens
    (max_src_len + max_tgt_len) × batch_size stays close to `max_tokens`.

    This is equivalent to the "batching by number of tokens" described in
    Vaswani et al. (2017) Section 5.1: "Each training batch contained a set
    of sentence pairs containing approximately 25000 source tokens and
    25000 target tokens."

    Algorithm:
        1. Sort examples by source length (reduces padding waste).
        2. Group consecutive examples into buckets where the running token
           count stays below `max_tokens`.
        3. Shuffle the bucket order each epoch for diversity.

    Parameters
    ----------
    dataset    : TranslationDataset
    max_tokens : int   Target tokens per batch (e.g. 4096 on a single GPU).
    shuffle    : bool  Shuffle bucket order each epoch.
    seed       : int   RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset    = dataset
        self.max_tokens = max_tokens
        self.shuffle    = shuffle
        self.rng        = random.Random(seed)

        # Pre-compute (src_len, tgt_len) for every example — used for bucketing
        self._lengths: List[Tuple[int, int]] = [
            (len(p.src_ids), len(p.tgt_ids))
            for p in dataset.pairs
        ]

        # Build initial sorted order (by source length)
        self._sorted_indices: List[int] = sorted(
            range(len(dataset)), key=lambda i: self._lengths[i][0]
        )

        self._batches: List[List[int]] = self._build_batches()
        logger.info(
            "TokenBucketSampler: %d examples → %d batches (max_tokens=%d)",
            len(dataset), len(self._batches), max_tokens,
        )

    def _build_batches(self) -> List[List[int]]:
        """
        Group sorted indices into variable-size batches.

        The batch size is determined dynamically: we track the current
        maximum source and target lengths in the forming batch, and flush
        when (max_src + max_tgt) × current_count would exceed max_tokens.
        This matches the PyTorch-Fairseq convention.
        """
        batches: List[List[int]] = []
        current_batch: List[int] = []
        max_src = max_tgt = 0

        for idx in self._sorted_indices:
            src_len, tgt_len = self._lengths[idx]

            # Compute the token budget if we added this example
            new_max_src = max(max_src, src_len)
            new_max_tgt = max(max_tgt, tgt_len)
            projected   = (new_max_src + new_max_tgt) * (len(current_batch) + 1)

            if current_batch and projected > self.max_tokens:
                # Flush current batch before it gets too large
                batches.append(current_batch)
                current_batch = [idx]
                max_src, max_tgt = src_len, tgt_len
            else:
                current_batch.append(idx)
                max_src, max_tgt = new_max_src, new_max_tgt

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches; optionally shuffle order each epoch."""
        if self.shuffle:
            self.rng.shuffle(self._batches)
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# Collation function
# ---------------------------------------------------------------------------

def collate_fn(
    pairs: List[TranslationPair],
    pad_id: int,
) -> Dict[str, Tensor]:
    """
    Convert a list of `TranslationPair` objects into a padded batch of tensors.

    Teacher-forcing convention:
        decoder_input  = tgt_ids[:-1]   (BOS … last_token)
        decoder_target = tgt_ids[1:]    (first_token … EOS)

    The loss is computed against `decoder_target` — we never try to predict
    what comes after EOS, and we never feed EOS as a decoder input during
    training (that is the model's job to predict).

    Parameters
    ----------
    pairs  : list of TranslationPair   Mini-batch examples (ragged sequences).
    pad_id : int                        PAD token id for right-padding.

    Returns
    -------
    dict with keys:
        'src'        : Tensor (batch, src_len)   Padded source ids.
        'tgt_in'     : Tensor (batch, tgt_len)   Decoder input (BOS … T-1).
        'tgt_out'    : Tensor (batch, tgt_len)   Decoder target (1 … EOS).
        'src_langs'  : list of str               Per-example source lang codes.
        'tgt_langs'  : list of str               Per-example target lang codes.
        'src_texts'  : list of str               Raw source strings (for BLEU).
        'tgt_texts'  : list of str               Raw target strings (for BLEU).
    """
    # Find max lengths in this batch for padding
    max_src = max(len(p.src_ids) for p in pairs)
    max_tgt = max(len(p.tgt_ids) - 1 for p in pairs)  # -1: split BOS/EOS

    src_padded:  List[List[int]] = []
    tgt_in_pad:  List[List[int]] = []
    tgt_out_pad: List[List[int]] = []

    for p in pairs:
        # Pad source to max_src
        src_seq = p.src_ids + [pad_id] * (max_src - len(p.src_ids))
        src_padded.append(src_seq)

        # Split target into decoder input/output
        tgt_in  = p.tgt_ids[:-1]   # BOS + tokens (no EOS fed to decoder)
        tgt_out = p.tgt_ids[1:]    # tokens + EOS (what we predict)

        # Pad to max_tgt
        tgt_in  = tgt_in  + [pad_id] * (max_tgt - len(tgt_in))
        tgt_out = tgt_out + [pad_id] * (max_tgt - len(tgt_out))

        tgt_in_pad.append(tgt_in)
        tgt_out_pad.append(tgt_out)

    return {
        "src":       torch.tensor(src_padded,  dtype=torch.long),
        "tgt_in":    torch.tensor(tgt_in_pad,  dtype=torch.long),
        "tgt_out":   torch.tensor(tgt_out_pad, dtype=torch.long),
        "src_langs": [p.src_lang for p in pairs],
        "tgt_langs": [p.tgt_lang for p in pairs],
        "src_texts": [p.src_text for p in pairs],
        "tgt_texts": [p.tgt_text for p in pairs],
    }


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset:    TranslationDataset,
    tokenizer:  MultilingualTokenizer,
    max_tokens: int  = 4096,
    shuffle:    bool = True,
    num_workers: int = 4,
    seed:       int  = 42,
) -> DataLoader:
    """
    Build a DataLoader with dynamic token-bucket batching.

    Parameters
    ----------
    dataset     : TranslationDataset
    tokenizer   : MultilingualTokenizer   Needed to get `pad_id` for collation.
    max_tokens  : int                     Token budget per batch.
    shuffle     : bool                    Shuffle batch order each epoch.
    num_workers : int                     DataLoader worker processes.
    seed        : int                     RNG seed.

    Returns
    -------
    DataLoader
        Each iteration yields a dict from `collate_fn`.
    """
    sampler = TokenBucketSampler(
        dataset, max_tokens=max_tokens, shuffle=shuffle, seed=seed
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,                  # TokenBucketSampler handles batching
        collate_fn=lambda batch: collate_fn(batch, pad_id=tokenizer.pad_id),
        num_workers=num_workers,
        pin_memory=True,                        # Faster CPU→GPU transfer
        persistent_workers=(num_workers > 0),   # Keep workers alive between epochs
    )


# ---------------------------------------------------------------------------
# Full data-pipeline convenience function
# ---------------------------------------------------------------------------

def build_datasets(
    config:       dict,
    tokenizer:    MultilingualTokenizer,
    max_examples: Optional[int] = None,
) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    Download, tokenise, and split all configured language pairs into
    train / val / test `TranslationDataset` objects.

    Parameters
    ----------
    config       : dict   Hydra config (expects config['Data'] sub-dict).
    tokenizer    : MultilingualTokenizer
    max_examples : int | None
        Cap on examples loaded per (pair, split).  Falls back to
        config['Data']['max_examples'] if set, then None (unlimited).
        Useful for smoke-tests without changing the yaml.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    data_cfg = config["Data"]
    max_len  = data_cfg["max_seq_length"]
    min_len  = data_cfg["min_seq_length"]

    # Resolve max_examples: explicit arg > config key > None (unlimited)
    if max_examples is None:
        max_examples = data_cfg.get("max_examples", None)

    all_train, all_val, all_test = [], [], []

    for pair_cfg in data_cfg["pairs"]:
        src_lang, tgt_lang = pair_cfg["src"], pair_cfg["tgt"]

        # Load each HF split separately (OPUS-100 provides predefined splits)
        for split, bucket in [("train", all_train), ("validation", all_val), ("test", all_test)]:
            try:
                raw = load_opus_pairs(
                    src_lang, tgt_lang,
                    split=split,
                    cache_dir=data_cfg.get("raw_dir"),
                    also_reverse=False,
                    max_examples=max_examples,
                    ratios=[data_cfg["train_ratio"], data_cfg["val_ratio"], data_cfg["test_ratio"]],
                )
            except Exception as exc:
                logger.warning("Could not load %s-%s [%s]: %s", src_lang, tgt_lang, split, exc)
                continue

            tokenised = tokenise_pairs(raw, tokenizer, max_length=max_len, min_length=min_len)
            bucket.extend(tokenised)

    # Shuffle train deterministically
    rng = random.Random(42)
    rng.shuffle(all_train)

    return (
        TranslationDataset(all_train),
        TranslationDataset(all_val),
        TranslationDataset(all_test),
    )