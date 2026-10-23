# Semantic parser

One extraction target, `semantic-document-v1`, is used for teacher annotation,
student supervision, and inference. The parser produces typed, source-grounded
documents; Hybrid Miner consumes those documents and the exported ontology
definitions. Rendering MeTTa is a downstream operation, not a second LLM task.

## Pipeline

```text
UTF-8 sentence file / dataset manifest / identified source records
    -> configured teacher or document-trained student
    -> catalog prompt + worked examples + optional constrained decoding
    -> representation normalization + exact quotation anchoring
    -> SemanticDocument + catalog validation
       |-> accepted document JSONL and explicit rejected-record JSONL
       |-> optional student pairs, from the same validated model response
       |-> exact ontology bundle for the miner
    -> independent miner ingestion -> emission/admission -> pattern mining
```

The teacher and student use the same target, prompt construction, active
ontology, source anchoring, and acceptance checks. They differ in model weights
and provider provenance. A model request generates the entities, assertions and
rules; deterministic code supplies source identity, parser provenance and
numeric offsets.

### Target and compatibility

`ReferenceSemanticParser.parse()` and `DistilledSemanticParser.parse()` return
`SemanticDocument`. `parse_document()` additionally accepts caller-assigned
document and corpus IDs. `build-dataset` and `export-documents` invoke the same
export function; neither selects an old assertion/rule extraction target.

`SemanticParseResult` remains a display/API representation for the existing
renderer and a reader for historical records. It is not independently generated,
a second dataset target, or a lossless miner interchange. Specifically:

- `generate_structured()` first generates a document, then derives a view.
- `compatibility.document_to_legacy_result()` makes that projection without any
  model call. Lexical arguments pass through the existing symbol normalization.
- `read_legacy_result()` validates saved old-format records without inference.
- `parse --format metta` renders a derived view. It does not replace the
  provenance-bearing JSONL or the miner's emitter.
- Old role/value labels cannot be converted into grounded document labels
  without recovering and auditing information they do not contain.

Empty accepted documents remain possible when no relation is supported.
Empty documents render no atoms. The historical `SemanticParseResult` model
still requires at least one assertion or rule.

This is an API migration, not a drop-in replacement for every historical call.
Previously `parse()` returned rendered atoms; it now returns the canonical
document. Call `generate_structured()` for the derived compatibility view or
use `parse --format metta` for rendering. Aliases affect that derived view only,
not canonical source evidence or entity identity.

`parse(sentence, context=...)` retains reference-only context: the sentence is
the extraction source and all offsets refer to it. Background can resolve an
unambiguous referent but cannot contribute additional assertions or source quotes.
A resolved participant without a literal mention in the sentence has a null span.
`parse_document()` accepts explicit document/corpus IDs and the same optional
context. For training such examples, retain `reference_context` in the pair or
producer metadata so the prompt can be reconstructed.

`build_document_prompt()` inspects an instance's selected catalog. The historical
static `build_prompt()` helper accepts an explicit `catalog=` and otherwise uses
the default catalog. Neither helper invokes a model.

Derived lexical MeTTa is not the miner's graph encoding. Rendering refuses
identity collisions and standalone factuality/negation scope that it cannot
represent faithfully; use canonical JSON for those documents. The optional
asserted-only view excludes rules and non-asserted standalone assertions without
changing the canonical document.

### Schema authority

The six independently versioned YAML catalogs in [ontology/](ontology/) define
known predicates and their ordered, typed signatures. The
[semantic-extraction manifest](ontology/bundles/semantic-extraction-v1.yaml)
selects the active bundle; by default it includes all six modules. The model
chooses relations within that supplied vocabulary. There is no autonomous
catalog router or hard-coded upper/discourse-only selection.

[Catalog loading](ontology/catalog.py) checks duplicate definitions, declared
types, arities, confidence bounds, and module versions.
[Bundle compilation](ontology/miner_bundle.py) exports these selected definitions
as the generic miner `OntologyBundle` format. The miner does not supply or require
these six project catalogs.

[predicate_schema.yaml](../configs/parser_config/predicate_schema.yaml) is
an active prompt input, as well as the authority for historical result validation.
Every predicate description is reused, with its ordered roles where applicable.
It is not blindly merged into the known catalog vocabulary:

| Legacy definition | Treatment in the shared document prompt |
| --- | --- |
| `Cause(cause, effect)` | Reviewed equivalent of `Causes(cause, effect)`; supplements its interpretation while catalog types and confidence bounds remain authoritative |
| Other fixed-arity predicates, such as `Near` and `During` | Lexical and ordered-participant guidance; use `ProvisionalRelation` unless a suitable active catalog predicate preserves the meaning |
| `Evaluation` | Reuse the fallback rationale, not the old role/value JSON shape or renderer-specific argument limits |
| Operators and structural containers | Not additional world-relation predicates; conjunction, implication and polarity use the document's explicit rule structure |

The prompt composer checks legacy arity against its ordered roles. A reviewed
catalog equivalent must have the same arity and argument-role order; drift fails
before model generation. A same-name collision without a reviewed equivalence
also fails rather than assuming semantic identity. These checks detect signature
drift; the equivalence itself remains a reviewed semantic decision.

A supported uncataloged relation retains a distinct provisional identifier and
typed arguments. No legacy relation is automatically approved merely because it
appears in `predicate_schema.yaml`. This preserves the paper's fixed, versioned
vocabulary and provisional-relation boundary without discarding existing guidance.

### Prompt composition and decoding

[Catalog prompt construction](semantic/catalog_prompt.py) assembles:

| Source | Responsibility |
| --- | --- |
| [catalog_document_prompt.yaml](../configs/parser_config/catalog_document_prompt.yaml) | Document output instructions, grounding, factuality, rules and provisional relations |
| [parser_prompt.yaml](../configs/parser_config/parser_prompt.yaml), `shared_extraction_guidance` | Reused interpretation guidance: voice, roles, coreference, negation, modality, ellipsis and conditions |
| [predicate_schema.yaml](../configs/parser_config/predicate_schema.yaml) | Every predicate description, checked ordered-role hints, and explicitly reviewed catalog equivalents; not extra known predicates |
| Active ontology modules | Authoritative known signatures and confidence bounds |
| [catalog_document_examples.jsonl](../configs/parser_config/catalog_document_examples.jsonl) | Worked typed, source-quoted document examples, including scoped rules |

The full historical target-specific template in `parser_prompt.yaml` is not an
active generation prompt. Its old output instructions cannot be appended to the
document prompt: the two JSON shapes conflict. Useful target-neutral guidance
is maintained explicitly in the shared section instead.

The numbered interpretation rules in the upstream template map to the document
pipeline as follows. This is a behavior mapping, not an extraction-quality score.

| Upstream rules | Document treatment |
| --- | --- |
| 1–16: predicate selection, complete participants, voice, ordered roles, purpose and persistent relations | Shared interpretation guidance, predicate descriptions, reviewed mappings and typed catalog/provisional arguments |
| 17: quantities and measurements | Exact source mentions retain numbers and units; symbol cleanup is restricted to derived rendering |
| 18: factuality | Explicit per-assertion factuality, separate from confidence and negation |
| 19–21: temporal/spatial modifiers and embedded clauses | Preserve core events and additional supported relations; uncataloged distinctions remain provisional |
| 22–25: missing information, ellipsis, references and negation | Conservative shared guidance, reference-only context, exact quotations, typed references and explicit scope |
| 26–31: output consistency, multiple propositions, confidence, source evidence and ambiguity | Document/signature validation, catalog confidence bounds, deterministic quotation anchoring and alternatives |
| 32–33: universal rules and predicate selection inside rules | Typed rule-local variables, signed conjunctive sides and source rule evidence; no execution of extracted rules |

Historical pronoun deletion and rule-repair routines remain in the historical
reader, not the active document path. They are not blindly applied to typed
graphs, where removing arguments can break signatures or change meaning.
Contract checks cannot establish correct pronoun resolution, completeness, or
factuality interpretation; those still require source-level model evaluation.

Prompt provenance includes a SHA-256 fingerprint of the assembled instructions,
examples and schema guidance, excluding the current source text. Changed
guidance invalidates stale training pairs/checkpoints even if the configured
prompt version was not changed.
The parser's `prompt_version` accessor reports this same active identity without
calling the model. The historical constructor version label is retained for
source compatibility; it does not select or override another extraction prompt.

The current composed prompt is `catalog-document-v14` plus that fingerprint.
Previously generated documents retain their recorded provenance. Training pairs
and checkpoints from an older prompt are deliberately not relabeled as current;
new supervision must be generated or separately reviewed by the parser team.

A prompt asks the model to follow the contract. Constrained decoding additionally
restricts the token choices using the generated JSON schema. The
[decoding schema](semantic/catalog_output_schema.py) is derived from the shared
contract and catalog signatures: it constrains predicate names, arity, argument
types and confidence ranges. It cannot establish semantic truth or correct
coreference. Post-generation validation is still required.

Ollama supports this JSON-schema request. `--require-structured-output` refuses
a backend without that capability before generation. Omitting that flag permits
post-validation-only use of other providers. The local Transformers student
does not currently implement grammar-constrained decoding.

### Normalization and evidence

[Document preparation](semantic/catalog_document_parser.py) normalizes
representation tokens, resolves source quotes, constructs the shared document,
and checks its contract and signatures. [Normalization](semantic/normalization.py) trims
predicate/type/enum whitespace and materializes optional condition fields.
It preserves source text, IDs, reference order, quoted evidence, alternatives,
and provisional relation meaning. It never repairs a wrong type by guessing.

[Source anchoring](semantic/catalog_source_anchors.py) maps an exact quote plus
its occurrence number to an offset. For example, `left` in `Mira left.` becomes
`[5,9)`. A quote not present in the source is rejected. Numeric span-bound
validation alone cannot determine which repeated mention the model meant;
the occurrence resolves that distinction.

The old symbol normalizer still serves rendered views, including punctuation,
Unicode, quantities and aliases. Applying lossy symbol normalization to
canonical source evidence would corrupt offsets or merge distinct mentions.
That is why document normalization and display-symbol normalization are
different operations, not competing parser targets.

### Shared validation and extracted rules

The parser imports `SemanticDocument`, its rule types, and signature checks from
the separately installed `hybrid-miner` package. These calls run in the parser
process; they do not start PeTTa or mine anything.

- Shared document validation checks structure, IDs/references, declared types,
  source bounds, assertion status, condition references and rule-variable scope.
- Signature checks compare predicates, arities and argument types with the
  supplied definitions.
- Parser-specific checks enforce exact quote anchoring, extraction factuality,
  selected catalog confidence ranges and interpretation policy.
- The independent miner validates its own input boundary when it later receives
  exported files. Another producer need not use this parser.

After emission, the miner's document-admission interface receives the original
documents along with facts, metadata and provenance. It quarantines complete
documents rather than stopping an otherwise valid batch. Its separate
`document-admission-v1` selection JSON identifies accepted source documents and
binds that selection to the artifacts. Pass it as `--admission` to the miner's
exact-matching and candidate-record commands. This is a downstream admission
receipt, not another LLM output target or a second parser path. The raw-graph
mining interface does not require this extraction-specific contract.

Document rules preserve explicit source implications with typed local variables,
nonempty antecedents/consequents, polarity and quotation evidence. Variables
cannot escape into standalone assertions or appear only in a consequent.
Extracting a stated rule is not executing it or claiming an S2 deduction.

Rule support is an engineering extension preserving useful existing-parser
capability within the document contract, not a claim that every field is
dictated verbatim by the papers. The installed miner release must expose these
rule types and the corresponding declarative rule-graph support. Its emitter
preserves rule scope, signed antecedents/consequents and source provenance;
it does not execute the rules or emit their conclusions as unconditional facts.
Mining yields structural fragments with complete source-rule context, not
permission to execute a shortened implication. The rule-graph encoding is
miner-owned interchange structure, not additional parser ontology vocabulary.

## Installation and model configuration

The miner-facing package requires Python 3.11 or newer:

```powershell
python -m pip install -e ".[miner,ollama]"
```

The `miner` dependency targets the independent repository's `v0.1.0` tag.
That tag must actually exist and include the required contract before a clean
installation or CI can resolve it. During unreleased coordinated development:

```powershell
python -m pip install -e ../hybrid-miner
```

Model providers and weights are configured in
[model_backend.yaml](../configs/parser_config/model_backend.yaml).
`--profile` precedes the subcommand. The `ollama_catalog` development profile
uses Qwen3-8B, temperature zero, thinking disabled and a document-sized generation
budget. Model scale alone does not establish teacher quality.

For a remote Ollama server accessed over SSH, keep the tunnel open and set the
host in the terminal executing the parser:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11435"
```

Never commit credentials or environment-specific connection details.

## Generate and inspect documents

Accepted input formats:

- UTF-8 text: one exact source sentence per line.
- Dataset manifest JSONL: `{"text":"Mira left.","source_dataset":"pilot","source_domain":"narrative"}`.
- Identified source JSONL: `{"document_id":"doc-1","corpus_id":"pilot","source_text":"Mira left."}`.

For sentence/manifest records, IDs are deterministic source-text hashes;
`--corpus-id` sets the default namespace. Manifest dataset/domain metadata is
retained in the producer metadata sidecar, whether or not student pairs are
requested. Identified records retain caller-supplied IDs.
Duplicate source IDs are rejected rather than overwritten.

Use fresh output names for each audited run:

```powershell
python -m parser.semantic --profile ollama_catalog build-dataset --input data/source.jsonl --output data/documents.jsonl --rejected-output data/rejected.jsonl --student-pairs-output data/student-pairs.jsonl --bundle-output data/ontology.bundle.json --require-structured-output
```

`export-documents` is an alternative command name for exactly this operation.
The older `parser.semantic.catalog_document_cli` entry point delegates to the
same implementation. `--output` always means documents; it never changes
meaning to training pairs. All outputs must have distinct, unused paths.

`--rejected-output` defaults to `<output-stem>.rejected.jsonl`.
`--metadata-output` defaults to `<output-stem>.metadata.jsonl`. Metadata is bound
to the document/corpus IDs and a SHA-256 of its exact source. It is not another
extraction target and is not required by miner ingestion. `--include-metta` adds
derived rendering there; `--asserted-only` filters that rendering, not the
canonical document. Unsupported rendering is recorded as `rendering_error`
without rejecting an otherwise valid document.

Exports report progress and checkpoint every 100 source records by default
(`--progress-every`). On interruption, generated records remain in `.partial`
artifacts with a `.checkpoint.json` manifest beside the document output. Repeat
the same command with `--resume`; source, prompt/model identity, options, and
artifact fingerprints must match before any further generation. Final files
are published only after processing completes. Existing completed outputs are
not overwritten. Resume recovers an unfinished publication as well.

Inspect the source beside the document: participant identity/order, negation,
factuality, conditional scope, rule restrictors and supported relations.
Acceptance proves contract compliance, not complete or correct extraction.
Rejected rows include their source line and reason; do not hide them from
acceptance metrics.

The bundle is a deterministic export of the selected vocabulary, not a model
response or a second parse. Supply `documents.jsonl` and `ontology.bundle.json`
to the independent miner's ingestion command. Generated experiment artifacts
need not be committed.

## Student data, training and inference

The paired label is the normalized quote-based `entities/assertions/rules`
payload that produced the document. Runtime IDs, model provenance and numeric
offsets are supplied deterministically again at inference; they are not a
second semantic prediction target. Pair provenance pins the teacher, prompt,
ontology versions, input and contract.

If labels were not saved during extraction, `convert-to-pairs` can reconstruct
them from validated documents and checks exact document round-trip equivalence:

```powershell
python -m parser.semantic convert-to-pairs --input data/documents.jsonl --output data/student-pairs.jsonl
python -m parser.semantic split-pairs --input data/student-pairs.jsonl --output-dir data/splits
python -m parser.semantic train-student --train-file data/splits/train.jsonl --val-file data/splits/validation.jsonl --method lora
```

Conversion automatically reads adjacent producer metadata. Supply
`--metadata-input` for a custom sidecar path. It rejects missing, extra, or
mismatched metadata records rather than assigning a domain to the wrong source.
`--include-confidence` retains the upstream first-assertion confidence metadata;
it does not filter or change the full label. `--max-per-text` is deprecated and
does not prune document labels.

The public `SemanticDatasetBuilder` retains `DistillationRecord` metadata
envelopes (`text`, teacher identity, dataset/domain and optional rendering).
`record.target` now contains the canonical document, and `record.document`
returns its validated contract model. Its writer exports document JSONL and an
adjacent metadata sidecar, not the historical nested-label JSON format.

[Dataset code](semantic/dataset.py) deduplicates input texts and checks that
identical input text does not overlap between splits. This does not establish
original-document-family separation. It also retains
complete-target tokenization, EOS supervision, prompt masking and padding
masking. Oversized or incompatible examples fail explicitly; no target is
truncated and no invalid label is silently skipped.

[Training](student/train.py) retains the existing LoRA/full optimization,
gradient accumulation, validation and safe precision selection.
[student_config.yaml](../configs/parser_config/student_config.yaml) now selects
the document target and bundle manifest. Upstream batch size 4 and gradient
accumulation 4 are retained. `max_length: auto` measures complete tokenized
prompt/label lengths before loading model weights; an integer supplies an
explicit ceiling. The measured maximum is reported and checked against the
tokenizer's declared context limit. This is not a GPU-memory benchmark or a
guarantee that a chosen batch size fits the available device. Its output
directory is `models/semantic_document_parser`; existing legacy checkpoints
are not overwritten by that default. An explicit training configuration can
select other model/hardware settings, but not another semantic target.

The existing `student_config.yaml` remains the sole default training
configuration. Training, corpus preparation at scale and student-quality
evaluation are separate operations; extraction/export does not execute them.

`e2e` retains direct `--sentences`, `--test-sentence`, cached validated splits,
and `--skip-training`. Fresh extraction continues with accepted pairs while
quarantining rejected sources; `--min-acceptance` supplies an explicit quality
threshold. Existing splits are checked for target compatibility and overlap
before use. `--force` creates a new child run instead of overwriting artifacts;
`--resume` recovers interrupted extraction. No default demonstration corpus is
silently generated when neither sources nor reusable splits are supplied.
Exact-target evaluation compares reconstructed documents on both sides, so
omitted optional defaults do not create false mismatches. It remains sensitive
to IDs, ordering and confidence; it is not a semantic-equivalence score.

After training:

```powershell
python -m parser.semantic distill --model-dir models/semantic_document_parser "Mira left."
python -m parser.semantic --profile semantic_document_student export-documents --input data/source.jsonl --output data/student.documents.jsonl --rejected-output data/student.rejected.jsonl
```

Checkpoint target, prompt fingerprint and ontology versions are checked before
loading weights. A checkpoint trained on a historical result target is not
automatically compatible with the document target. Its evaluation results do
not establish quality on a different output contract.
Deployment claims require a trained document checkpoint and held-out
semantic evaluation. Successful teacher generation does not establish that.

`e2e` composes extraction, text-deduplicated splitting, the existing trainer and
document inference in a fresh explicit run directory. Its exact-target comparison
is ID-sensitive, not a semantic-equivalence benchmark.

## Source map

| File | Responsibility |
| --- | --- |
| [semantic_parser.py](semantic/semantic_parser.py) | Shared teacher/student document generation and historical reader |
| [backends.py](semantic/backends.py), [model_config.py](semantic/model_config.py) | Existing model providers, profiles and local execution |
| [output_cleanup.py](semantic/output_cleanup.py) | Response-envelope cleanup, not semantic repair |
| [catalog_document_parser.py](semantic/catalog_document_parser.py) | Document preparation and compatibility constructor |
| [catalog_document_export.py](semantic/catalog_document_export.py) | One streaming exporter for documents, labels, rejections and bundle |
| [catalog_student.py](semantic/catalog_student.py) | Label reconstruction and checkpoint compatibility |
| [compatibility.py](semantic/compatibility.py) | Derived legacy display/API projection only |
| [schema.py](semantic/schema.py), [metta_renderer.py](semantic/metta_renderer.py) | Historical/result-view types and deterministic rendering |
| [cli.py](semantic/cli.py) | Production command interface |
| [student_prompt.py](student/student_prompt.py) | Same catalog prompt for student training/inference |

## Verification

Run regression coverage and then inspect actual provider output:

```powershell
python -m pytest tests/unit tests/integration/test_catalog_document_parser.py tests/integration/test_catalog_document_export.py tests/integration/test_catalog_bundle.py -q
```

Unit/integration tests cover deterministic behavior. The production commands above
are the manual checks for live model behavior. Audit rejection reasons and source
meaning separately from structural acceptance. Report inference/runtime settings,
data, denominators and measured timings when comparing models; no speed or
semantic-quality claim follows merely from passing tests.
