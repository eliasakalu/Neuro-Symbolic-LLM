"""One semantic-document CLI for teacher extraction, distillation and inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from parser.semantic.dataset import (
    split_pairs,
    structured_to_pairs,
    validate_student_pair,
    verify_split_overlap,
    write_pairs_jsonl,
)
from parser.semantic.model_config import build_reference_semantic_parser
from parser.semantic.semantic_parser import (
    DistilledSemanticParser,
    ModelGenerationError,
    SemanticParseError,
)


def add_document_export_arguments(arguments: argparse.ArgumentParser) -> None:
    """Register the same document export options for both entry points."""
    arguments.add_argument("--input", required=True, type=Path)
    arguments.add_argument("--output", required=True, type=Path)
    arguments.add_argument("--rejected-output", required=True, type=Path)
    arguments.add_argument("--student-pairs-output", type=Path)
    arguments.add_argument(
        "--corpus-id", help="namespace for sentence or manifest inputs"
    )
    add_document_target_arguments(arguments)


def add_document_target_arguments(arguments: argparse.ArgumentParser) -> None:
    """Share catalog and decoding configuration across dataset and export commands."""
    arguments.add_argument(
        "--bundle-output",
        type=Path,
        help="export the exact ontology bundle used for document validation",
    )
    arguments.add_argument("--model-revision")
    arguments.add_argument(
        "--require-structured-output",
        action="store_true",
        help="fail before generation if the backend cannot constrain JSON-schema decoding",
    )
    ontology = arguments.add_mutually_exclusive_group()
    ontology.add_argument("--ontology-bundle", type=Path)
    ontology.add_argument(
        "--ontology-module", action="append", dest="ontology_modules", metavar="NAME"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """All generation commands use the semantic-document-v1 target."""
    parser = argparse.ArgumentParser(prog="semantic-parser")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-dataset", "export-documents"):
        command = commands.add_parser(
            name, help="generate validated documents and optional student pairs"
        )
        add_document_export_arguments(command)
        command.add_argument(
            "--target",
            choices=("semantic-document-v1",),
            default="semantic-document-v1",
        )
    parse = commands.add_parser(
        "parse", help="parse exact source text to a semantic document"
    )
    parse.add_argument("sentence")
    parse.add_argument("--context", default="")
    parse.add_argument("--format", choices=("json", "metta"), default="json")
    convert = commands.add_parser(
        "convert-to-pairs", help="recover document-target student pairs"
    )
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    split = commands.add_parser("split-pairs")
    split.add_argument("--input", required=True, type=Path)
    split.add_argument("--output-dir", required=True, type=Path)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train-student")
    training.add_argument("--train-file", required=True, type=Path)
    training.add_argument("--val-file", type=Path)
    training.add_argument("--training-config", type=Path)
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--method", choices=("lora", "full"), default="full")
    student = commands.add_parser(
        "distill", help="infer the same target with a document-trained student"
    )
    student.add_argument("sentence")
    student.add_argument(
        "--model-dir", type=Path, default=Path("models/semantic_document_parser")
    )
    student.add_argument("--format", choices=("json", "metta"), default="json")
    e2e = commands.add_parser(
        "e2e", help="extract, split, train and evaluate one document target"
    )
    e2e.add_argument("--input", required=True, type=Path)
    e2e.add_argument("--output-dir", required=True, type=Path)
    e2e.add_argument("--training-config", type=Path)
    e2e.add_argument("--method", choices=("lora", "full"), default="lora")
    return parser


def _print_document(document, output_format: str) -> None:
    if output_format == "json":
        print(document.canonical_json())
    elif document.assertions or document.rules:
        from parser.semantic.compatibility import document_to_legacy_result
        from parser.semantic.semantic_parser import ReferenceSemanticParser

        view = document_to_legacy_result(document)
        for atom in ReferenceSemanticParser.render_metta(view):
            print(atom)


def _run_parse(arguments: argparse.Namespace) -> None:
    parser = build_reference_semantic_parser(
        arguments.config, profile_name=arguments.profile
    )
    _print_document(
        parser.parse(arguments.sentence, context=arguments.context), arguments.format
    )


def _run_convert_to_pairs(arguments: argparse.Namespace) -> None:
    pairs = structured_to_pairs(str(arguments.input), str(arguments.output))
    print(f"Student pairs: {len(pairs)} -> {arguments.output}")


def _run_split_pairs(arguments: argparse.Namespace) -> None:
    """Split student pairs into train, validation, and test sets."""

    input_path: Path = arguments.input
    output_dir: Path = arguments.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    if (
        len({path.resolve() for path in (input_path, train_path, val_path, test_path)})
        != 4
    ):
        raise ValueError("Input and split-output paths must differ")
    for path in (train_path, val_path, test_path):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing split: {path}")

    pairs = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at " f"{input_path}:{line_number}: " f"{error.msg}"
                ) from error
            if not isinstance(pair, dict) or not isinstance(pair.get("input", ""), str):
                raise ValueError(
                    f"Invalid pair at "
                    f"{input_path}:{line_number}: "
                    "expected an object with a string input"
                )

            try:
                validate_student_pair(pair)
            except ValueError as error:
                raise ValueError(
                    f"Invalid pair at {input_path}:{line_number}: {error}"
                ) from error
            pairs.append(pair)

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        test_ratio=arguments.test_ratio,
        seed=arguments.seed,
    )

    verify_split_overlap(train_pairs, val_pairs, test_pairs)

    write_pairs_jsonl(train_pairs, train_path)
    write_pairs_jsonl(val_pairs, val_path)
    write_pairs_jsonl(test_pairs, test_path)

    print(f"Train:      {len(train_pairs)} -> {train_path}")
    print(f"Validation: {len(val_pairs)} -> {val_path}")
    print(f"Test:       {len(test_pairs)} -> {test_path}")


def _run_train_student(arguments: argparse.Namespace) -> None:
    """Train student model."""
    from parser.student.train import train

    print("Training student model...")
    print(f"  Training file: {arguments.train_file}")
    print(f"  Method: {arguments.method}")
    print(f"  Output: {arguments.output_dir}")

    training_options = {}
    if arguments.training_config is not None:
        training_options["config_path"] = str(arguments.training_config)
    train(
        train_file=str(arguments.train_file),
        val_file=str(arguments.val_file) if arguments.val_file else None,
        output_dir=str(arguments.output_dir) if arguments.output_dir else None,
        method=arguments.method,
        **training_options,
    )


def _checkpoint_error(model_dir: Path) -> str | None:
    """Check local checkpoint files without loading model weights into memory."""
    if not model_dir.is_dir():
        return "model directory is missing or is not a directory"

    try:
        from transformers import AutoConfig, AutoTokenizer

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

        if not isinstance(config, dict) or not config:
            return "config.json must contain a model configuration object"
        AutoConfig.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )

        AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )

        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ):
            candidate = model_dir / name

            if not candidate.exists():
                continue

            if name.endswith(".index.json"):
                index = json.loads(candidate.read_text(encoding="utf-8"))

                weight_map = (
                    index.get("weight_map") if isinstance(index, dict) else None
                )

                if not isinstance(weight_map, dict) or not weight_map:
                    return f"{name} has no valid weight map"

                shards = list(weight_map.values())

                if any(not isinstance(shard, str) or not shard for shard in shards):
                    return f"{name} contains invalid shard names"

                files = [model_dir / shard for shard in set(shards)]

            else:
                files = [candidate]

            for path in files:
                if not path.is_file():
                    return f"missing weight file: " f"{path.name}"

                with path.open("rb") as file:
                    if not file.read(1):
                        return f"empty weight file: " f"{path.name}"

            return None

    except Exception as error:
        return f"unreadable checkpoint: {error}"

    return "no model weights or weight index found"


def _require_checkpoint(model_dir: Path) -> None:
    error = _checkpoint_error(model_dir)

    if error is not None:
        raise ValueError(
            "A complete model checkpoint is required " f"at {model_dir}: {error}"
        )


def _run_distill(arguments: argparse.Namespace) -> None:
    _require_checkpoint(arguments.model_dir)
    parser = DistilledSemanticParser.from_pretrained(str(arguments.model_dir))
    _print_document(parser.parse(arguments.sentence), arguments.format)


def _run_e2e(arguments: argparse.Namespace) -> int:
    """Use an explicit fresh run directory, never reuse stale labels/checkpoints."""
    from parser.semantic.catalog_document_cli import run_export
    from parser.semantic.catalog_student import document_target
    from parser.student.train import train

    root = arguments.output_dir
    if root.exists():
        raise ValueError(
            "e2e requires a new run directory; existing artifacts are never overwritten"
        )
    documents, pairs = root / "documents.jsonl", root / "pairs.jsonl"
    export_args = argparse.Namespace(
        config=arguments.config,
        profile=arguments.profile,
        input=arguments.input,
        output=documents,
        rejected_output=root / "rejected.jsonl",
        student_pairs_output=pairs,
        bundle_output=root / "ontology.bundle.json",
        ontology_bundle=None,
        ontology_modules=None,
        model_revision=None,
        require_structured_output=False,
        corpus_id=None,
    )
    if run_export(export_args):
        raise ValueError("Extraction rejected sources; audit the run before training")
    _run_split_pairs(
        argparse.Namespace(
            input=pairs,
            output_dir=root / "splits",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
    )
    train_file, val_file, test_file = (
        root / "splits" / name
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    )
    if not train_file.stat().st_size or not test_file.stat().st_size:
        raise ValueError("Need nonempty source-disjoint training and test sets")
    options = (
        {"config_path": str(arguments.training_config)}
        if arguments.training_config
        else {}
    )
    model_dir = train(
        train_file=str(train_file),
        val_file=str(val_file),
        method=arguments.method,
        output_dir=str(root / "student"),
        **options,
    )
    student = DistilledSemanticParser.from_pretrained(model_dir)
    valid = exact = total = 0
    with test_file.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            pair = json.loads(line)
            validate_student_pair(pair)
            total += 1
            try:
                document = student.parse(pair["input"])
                valid += 1
                exact += int(document_target(document) == json.loads(pair["label"]))
                print(document.canonical_json())
            except (ValueError, RuntimeError) as error:
                print(f"Prediction rejected: {error}", file=sys.stderr)
    print(
        f"Validated documents: {valid}/{total}; exact target matches: {exact}/{total}"
    )
    print(
        "Exact target matching is ID-sensitive; this is not a semantic equivalence metric."
    )
    return 0 if valid == total else 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command in ("build-dataset", "export-documents"):
            from parser.semantic.catalog_document_cli import run_export

            return run_export(arguments)
        handlers = {
            "parse": _run_parse,
            "distill": _run_distill,
            "convert-to-pairs": _run_convert_to_pairs,
            "split-pairs": _run_split_pairs,
            "train-student": _run_train_student,
        }
        if arguments.command == "e2e":
            return _run_e2e(arguments)
        handlers[arguments.command](arguments)
    except (ModelGenerationError, OSError, SemanticParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
