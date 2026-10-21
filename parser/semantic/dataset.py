"""Canonical document datasets and masked-loss student tokenization."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from hybrid_miner.contracts import SemanticDocument
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_student import (
    make_student_pair,
    validate_catalog_student_pair,
)
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
)

if TYPE_CHECKING:
    from parser.ontology.catalog import OntologyCatalog

# ── Dataset records ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One source sentence rejected by the reference pipeline."""

    text: str
    error: str


# ── Dataset generator ──────────────────────────────────────────────────────────


class SemanticDatasetBuilder:
    """Build validated distillation data with a ReferenceSemanticParser."""

    def __init__(
        self,
        parser: ReferenceSemanticParser,
        *,
        include_metta: bool = False,
        asserted_only: bool = False,
    ) -> None:
        """Store the parser and dataset-generation policy."""
        self._parser = parser
        if include_metta or asserted_only:
            raise ValueError(
                "Canonical datasets preserve factuality and rules; render or filter downstream"
            )

    def generate(
        self,
        sentences: Iterable[str],
        *,
        source_dataset: str | None = None,
        source_domain: str | None = None,
    ) -> tuple[list[SemanticDocument], list[RejectedRecord]]:
        """Parse sentences and separate accepted from rejected examples."""
        accepted = []
        rejected = []

        for sentence in sentences:
            text = sentence.strip()
            if not text:
                rejected.append(
                    RejectedRecord(
                        text=sentence,
                        error="The sentence cannot be empty",
                    )
                )
                continue

            try:
                document = self._parser.parse(sentence)
                if source_dataset is not None:
                    document = SemanticDocument.model_validate(
                        {
                            **document.model_dump(mode="json"),
                            "corpus_id": source_dataset,
                        }
                    )
                accepted.append(document)
            except (ModelGenerationError, SemanticParseError, ValueError) as error:
                rejected.append(RejectedRecord(text=sentence, error=str(error)))

        return accepted, rejected

    @staticmethod
    def write_jsonl(
        records: Iterable[SemanticDocument],
        output_path: str | Path,
    ) -> None:
        """Write accepted records as UTF-8 JSON Lines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(record.canonical_json() + "\n")

    @staticmethod
    def write_rejected_jsonl(
        records: Iterable[RejectedRecord],
        output_path: str | Path,
    ) -> None:
        """Write rejected records as UTF-8 JSON Lines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


# ── Student Dataset for training ─────────────────────────────────────────────


def validate_student_pair(pair: Any, *, catalog: OntologyCatalog | None = None) -> None:
    """Reject malformed input/label records before tokenization or splitting."""
    validate_catalog_student_pair(
        pair, catalog=catalog if catalog is not None else load_ontology_catalog()
    )


class StudentDataset(Dataset):
    """Prepares (input, json_label) pairs for student model training."""

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        max_length: int = 16384,
        *,
        catalog: OntologyCatalog | None = None,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("Student training requires an EOS token")
        self.examples = []
        self.oversized = 0
        active_catalog = catalog if catalog is not None else load_ontology_catalog()

        for pair_index, pair in enumerate(pairs, start=1):
            prompt_text, json_label, _ = validate_catalog_student_pair(
                pair, catalog=active_catalog
            )
            # Encode the boundary separately: prompt tokens must match inference,
            # and every target token must contribute to the loss.
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(json_label, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                raise ValueError(
                    f"Catalog pair {pair_index} needs {len(input_ids)} tokens; "
                    f"max_length is {max_length}. Targets are never truncated."
                )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


@dataclass(frozen=True)
class StudentBatchCollator:
    """Right-pad inputs while excluding padding from attention and loss."""

    pad_token_id: int

    def __call__(
        self, examples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        padding = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        return {
            key: pad_sequence(
                [example[key] for example in examples],
                batch_first=True,
                padding_value=value,
            )
            for key, value in padding.items()
        }


# ── Converter: Structured → Pairs (for student training) ─────────────────────


def structured_to_pairs(
    input_file: str,
    output_file: str,
    max_per_text: int = 1,
    include_confidence: bool = False,
    *,
    catalog: OntologyCatalog | None = None,
) -> list[dict]:
    """Recover the same training target from documents; reject historical labels."""
    source, destination = Path(input_file), Path(output_file)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("Input and output paths must differ")
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing output: {destination}")
    active_catalog = catalog if catalog is not None else load_ontology_catalog()
    pairs = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                document = SemanticDocument.model_validate_json(line)
                pair = make_student_pair(document)
                _, _, reconstructed = validate_catalog_student_pair(
                    pair, catalog=active_catalog
                )
                if reconstructed.canonical_json() != document.canonical_json():
                    raise ValueError("Conversion would change the document")
                pairs.append(pair)
            except ValueError as error:
                raise ValueError(
                    f"Invalid document at {source}:{line_number}: {error}"
                ) from error
    write_pairs_jsonl(pairs, destination)
    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deduplicate inputs, shuffle, and split using largest-remainder rounding."""

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("Split ratios must be finite numbers between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Train, validation and test ratios must sum to 1.0")

    unique_pairs = []
    seen = set()

    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict) or not isinstance(pair.get("input", ""), str):
            raise ValueError(f"Pair {index} must be an object with a string input")
        text = pair.get("input", "").strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        unique_pairs.append(pair)

    rng = random.Random(seed)
    rng.shuffle(unique_pairs)

    total = len(unique_pairs)
    # Normalize tolerated floating-point error before assigning all records.
    sizes = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [math.floor(size) for size in sizes]
    order = sorted(range(3), key=lambda i: sizes[i] - counts[i], reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    train_end = counts[0]
    val_end = train_end + counts[1]

    train_pairs = unique_pairs[:train_end]
    val_pairs = unique_pairs[train_end:val_end]
    test_pairs = unique_pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


def verify_split_overlap(
    train_pairs: list[dict],
    val_pairs: list[dict],
    test_pairs: list[dict],
) -> None:
    """Ensure the same input text does not appear across dataset splits."""

    train_texts = {pair["input"].strip() for pair in train_pairs}
    val_texts = {pair["input"].strip() for pair in val_pairs}
    test_texts = {pair["input"].strip() for pair in test_pairs}

    if not train_texts.isdisjoint(val_texts):
        raise ValueError("Train and validation sets overlap")

    if not train_texts.isdisjoint(test_texts):
        raise ValueError("Train and test sets overlap")

    if not val_texts.isdisjoint(test_texts):
        raise ValueError("Validation and test sets overlap")


def write_pairs_jsonl(pairs: list[dict], output_file: str | Path) -> None:
    """Write student input-label pairs to JSONL"""

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
