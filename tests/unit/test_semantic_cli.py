"""Production commands share one document exporter and one training target."""

import json

import pytest

from parser.semantic import cli, model_config
from parser.semantic.backends import CallableBackend
from tests.semantic_fixtures import pair, target


@pytest.fixture
def backend(monkeypatch):
    calls = []

    def generate(*, prompt, model):
        calls.append(prompt)
        text = prompt.rsplit("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        return json.dumps(target(text))

    monkeypatch.setattr(
        model_config, "create_backend", lambda profile: CallableBackend(generate)
    )
    return calls


def test_parse_defaults_to_document_json(backend, capsys):
    assert cli.main(["parse", "Mira left."]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["contract_version"] == "semantic-document-v1"
    assert result["assertions"][0]["predicate"] == "Event"
    assert len(backend) == 1


def test_parse_metta_is_only_a_derived_view(backend, capsys):
    assert cli.main(["parse", "Mira left.", "--format", "metta"]) == 0
    assert "Evaluation" in capsys.readouterr().out
    assert len(backend) == 1


@pytest.mark.parametrize("command", ["build-dataset", "export-documents"])
def test_dataset_commands_export_the_same_target(tmp_path, backend, command):
    source, docs, labels = (
        tmp_path / "sources.txt",
        tmp_path / "docs.jsonl",
        tmp_path / "pairs.jsonl",
    )
    source.write_text("Mira left.\n", encoding="utf-8")
    assert (
        cli.main(
            [
                command,
                "--input",
                str(source),
                "--output",
                str(docs),
                "--student-pairs-output",
                str(labels),
                "--rejected-output",
                str(tmp_path / "rejected.jsonl"),
            ]
        )
        == 0
    )
    document = json.loads(docs.read_text(encoding="utf-8"))
    student_pair = json.loads(labels.read_text(encoding="utf-8"))
    assert (
        document["contract_version"]
        == student_pair["contract_version"]
        == "semantic-document-v1"
    )
    assert len(backend) == 1


def test_manifest_source_metadata_survives_into_training_pair(tmp_path, backend):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "text": "Mira left.",
                "source_dataset": "pilot",
                "source_domain": "narrative",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    docs, labels = tmp_path / "docs.jsonl", tmp_path / "pairs.jsonl"
    assert (
        cli.main(
            [
                "build-dataset",
                "--input",
                str(source),
                "--output",
                str(docs),
                "--student-pairs-output",
                str(labels),
                "--rejected-output",
                str(tmp_path / "rejected.jsonl"),
            ]
        )
        == 0
    )
    record = json.loads(labels.read_text(encoding="utf-8"))
    assert record["source_dataset"] == "pilot"
    assert record["source_domain"] == "narrative"
    assert json.loads(docs.read_text(encoding="utf-8"))["corpus_id"] == "pilot"


@pytest.mark.parametrize("collision", ["input", "rejected", "labels"])
def test_export_refuses_path_collisions_without_data_loss(tmp_path, backend, collision):
    source, docs = tmp_path / "source.txt", tmp_path / "docs.jsonl"
    source.write_text("Mira left.\n", encoding="utf-8")
    rejected, labels = tmp_path / "rejected.jsonl", tmp_path / "pairs.jsonl"
    if collision == "input":
        docs = source
    elif collision == "rejected":
        rejected = docs
    else:
        labels = docs
    assert (
        cli.main(
            [
                "build-dataset",
                "--input",
                str(source),
                "--output",
                str(docs),
                "--student-pairs-output",
                str(labels),
                "--rejected-output",
                str(rejected),
            ]
        )
        == 1
    )
    assert source.read_text(encoding="utf-8") == "Mira left.\n"
    assert backend == []


def test_help_supports_windows_encoding():
    assert cli.build_argument_parser().format_help().encode("cp1252")


def test_legacy_target_is_not_an_active_choice():
    with pytest.raises(SystemExit):
        cli.build_argument_parser().parse_args(
            ["build-dataset", "--target", "legacy-semantic-v1"]
        )


def test_split_validates_labels_and_writes_disjoint_sources(tmp_path):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        "".join(json.dumps(pair(f"Source {i}.")) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    output = tmp_path / "splits"
    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(output)])
        == 0
    )
    splits = [
        [
            json.loads(line)
            for line in (output / name).read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    ]
    assert [len(items) for items in splits] == [8, 1, 1]
    from parser.semantic.dataset import verify_split_overlap

    verify_split_overlap(*splits)


@pytest.mark.parametrize("record", [None, [], {"input": "Dog", "label": "{}"}])
def test_split_refuses_old_or_invalid_labels_before_writing(tmp_path, record):
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(record), encoding="utf-8")
    assert (
        cli.main(
            [
                "split-pairs",
                "--input",
                str(source),
                "--output-dir",
                str(tmp_path / "splits"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "splits").exists()


def test_e2e_never_reuses_an_existing_run(tmp_path):
    assert (
        cli.main(
            [
                "e2e",
                "--input",
                str(tmp_path / "source.txt"),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_train_command_passes_existing_training_controls(monkeypatch, tmp_path):
    from parser.student import train as training

    options = []
    monkeypatch.setattr(training, "train", lambda **kwargs: options.append(kwargs))
    assert (
        cli.main(
            [
                "train-student",
                "--train-file",
                str(tmp_path / "train.jsonl"),
                "--val-file",
                str(tmp_path / "val.jsonl"),
                "--method",
                "lora",
            ]
        )
        == 0
    )
    assert options[0]["method"] == "lora"
    assert options[0]["val_file"].endswith("val.jsonl")
    assert options[0]["output_dir"] is None


def write_checkpoint(path):
    """Save a tiny usable checkpoint without network access."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "dog": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=2, n_embd=8, n_layer=1, n_head=1, n_positions=16)
    )
    model.save_pretrained(path)


@pytest.mark.parametrize(
    "damage",
    [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "empty_weights",
        "bad_config",
        "bad_tokenizer",
    ],
)
def test_checkpoint_rejects_missing_or_unreadable_files(tmp_path, damage):
    write_checkpoint(tmp_path)
    if damage == "empty_weights":
        (tmp_path / "model.safetensors").write_bytes(b"")
    elif damage == "bad_config":
        (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    elif damage == "bad_tokenizer":
        (tmp_path / "tokenizer.json").write_text("null", encoding="utf-8")
    else:
        (tmp_path / damage).unlink()
    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_requires_every_indexed_shard(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").rename(tmp_path / "shard-1.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight_a": "shard-1.safetensors",
                    "weight_b": "shard-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    assert "shard-2.safetensors" in cli._checkpoint_error(tmp_path)
    (tmp_path / "shard-2.safetensors").write_bytes(
        (tmp_path / "shard-1.safetensors").read_bytes()
    )
    assert cli._checkpoint_error(tmp_path) is None


@pytest.mark.parametrize(
    "index", [None, {}, {"weight_map": {}}, {"weight_map": {"a": None}}]
)
def test_checkpoint_rejects_malformed_index(tmp_path, index):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_accepts_pytorch_weights(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    import torch

    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")
    assert cli._checkpoint_error(tmp_path) is None


def test_distill_requires_real_checkpoint(tmp_path):
    assert cli.main(["distill", "--model-dir", str(tmp_path), "Mira left."]) == 1
