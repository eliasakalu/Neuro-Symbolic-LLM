"""parser/semantic/semantic_parser.py

Structured natural-language semantic parsers.

ReferenceSemanticParser uses a large model (teacher) for offline annotation.
DistilledSemanticParser uses a saved fine-tuned model (student) for inference.

Both generate the same document target; rendering is a derived view.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import yaml
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser.grammar.atomese import LinkAtom
from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.backends import ModelBackend
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.metta_renderer import (
    MettaRenderError,
)
from parser.semantic.metta_renderer import (
    render_metta as render_semantic_result,
)
from parser.semantic.metta_renderer import (
    validate_rendered_metta as parse_rendered_metta,
)
from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_semantic_result,
)
from parser.semantic.output_cleanup import clean_model_output
from parser.semantic.schema import SemanticParseResult

if TYPE_CHECKING:
    from hybrid_miner.contracts.semantic_document import SemanticDocument


class SemanticParseError(ValueError):
    """Raised when model output cannot be converted into valid semantics."""


class ModelGenerationError(RuntimeError):
    """Raised when a model backend cannot generate usable output."""


# ── Configuration ──────────────────────────────────────────────────────────────

_PARSER_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "parser_prompt.yaml"
)

_PREDICATE_SCHEMA_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "predicate_schema.yaml"
)


def _load_predicate_schemas() -> dict[str, dict[str, object]]:
    """Load predicate schemas and reject invalid configuration early."""
    with _PREDICATE_SCHEMA_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Predicate schema must be a YAML mapping")
    predicates = config.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise ValueError("Predicate schema must define a nonempty 'predicates' mapping")

    for name, schema in predicates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Predicate names must be nonempty strings")
        if not isinstance(schema, dict):
            raise ValueError(f"Schema for {name!r} must be a mapping")
        description = schema.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Schema for {name!r} requires a description")

        if schema.get("variable_arity") is True:
            minimum = schema.get("min_arity")
            maximum = schema.get("max_arity")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
                raise ValueError(f"Invalid min_arity for {name!r}")
            if maximum is not None and (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < minimum
            ):
                raise ValueError(f"Invalid max_arity for {name!r}")
            continue

        arity = schema.get("arity")
        roles = schema.get("roles")
        if not isinstance(arity, int) or isinstance(arity, bool) or arity < 1:
            raise ValueError(f"Invalid arity for {name!r}")
        if (
            not isinstance(roles, list)
            or len(roles) != arity
            or any(not isinstance(role, str) or not role.strip() for role in roles)
        ):
            raise ValueError(f"Schema roles for {name!r} must match its arity")

    evaluation = predicates.get("Evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("fallback") is not True:
        raise ValueError("Evaluation must be defined with fallback=true")
    return predicates


PREDICATE_SCHEMAS = _load_predicate_schemas()

# Historical reader vocabulary; active generation uses the ontology catalogs.
ALLOWED_PREDICATES = ", ".join(sorted(PREDICATE_SCHEMAS))


def _build_predicate_guide() -> str:
    """Describe predicate roles directly from the validated YAML schema."""
    lines = []
    for name, schema in PREDICATE_SCHEMAS.items():
        description = schema["description"]
        if schema.get("variable_arity") is True:
            roles = "semantic roles appropriate to the relation"
        else:
            roles = ", ".join(cast(list[str], schema["roles"]))
        lines.append(f"- {name}({roles}): {description}")
    return "\n".join(lines)


PREDICATE_GUIDE = _build_predicate_guide()


def _iter_all_assertions(result: SemanticParseResult):
    """Yield standalone assertions and assertions contained in rules."""
    yield from result.assertions

    for rule in result.rules:
        yield from rule.antecedents
        yield from rule.consequents


@dataclass(frozen=True, slots=True)
class SemanticParserConfig:
    """Configuration shared by both semantic parser roles."""

    model_name: str
    # Retained constructor compatibility. The active extraction prompt's
    # identity is derived from its assembled guidance, not this legacy label.
    prompt_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be empty")


# ── Base Parser ────────────────────────────────────────────────────────────────


class _BaseSemanticParser:
    """Shared structured semantic extraction pipeline.

    Supports both:
    - Reference (teacher): uses backend (external API) to generate
    - Distilled (student): uses saved model from disk
    """

    parser_role = "semantic"
    max_new_tokens = 256
    num_beams = 4

    def __init__(
        self,
        *,
        backend: ModelBackend | None = None,
        config: SemanticParserConfig,
        model: Any = None,
        tokenizer: Any = None,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
        require_structured_output: bool = False,
    ) -> None:
        """Create a parser with either backend or a loaded model."""
        self._backend = backend
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._catalog = catalog
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision cannot be empty")
        self._model_revision = model_revision
        self._require_structured_output = require_structured_output

    @property
    def provider_name(self) -> str:
        if self._backend is not None:
            return self._backend.provider_name
        return "distilled-local"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def prompt_version(self) -> str:
        """Return the active prompt identity without invoking the model."""
        catalog = (
            self._catalog if self._catalog is not None else load_ontology_catalog()
        )
        return build_catalog_document_prompt("prompt identity", catalog=catalog).version

    @staticmethod
    def build_prompt(sentence: str, context: str = "") -> str:
        """Build the same catalog-constrained prompt used for document labels."""
        if not sentence.strip():
            raise ValueError("The sentence cannot be empty")
        source_text = context + "\n" + sentence if context else sentence
        return build_catalog_document_prompt(
            source_text, catalog=load_ontology_catalog()
        ).text

    @staticmethod
    def clean_model_output(output: str) -> str:
        """Unwrap JSON presentation without altering decoded semantic values."""
        return clean_model_output(output)

    @staticmethod
    def repair_known_predicate_contract(
        result: SemanticParseResult,
    ) -> SemanticParseResult:
        """Repair deterministic contract fields for known predicates."""
        repaired = result.model_copy(deep=True)

        for assertion in _iter_all_assertions(repaired):
            if (
                assertion.predicate in PREDICATE_SCHEMAS
                and assertion.predicate != "Evaluation"
            ):
                assertion.relation = None
                assertion.fallback = False

        return repaired

    @staticmethod
    def repair_ambiguous_pronouns(
        result: SemanticParseResult,
    ) -> SemanticParseResult:
        """Repair or remove unresolved referential pronouns safely.

        Standalone assertions preserve the existing behavior: unresolved
        referential pronouns are removed when possible.

        Inside rules, a pronoun may be resolved only when the rule has exactly
        one clear generic participant such as ``someone``, ``something``, or a
        MeTTa variable. Otherwise the rule is rejected rather than guessed.

        Expletive ``it`` in weather expressions is preserved.
        """
        repaired = result.model_copy(deep=True)

        pronouns = {
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
        }

        expletive_it_relations = {
            "rain",
            "snow",
            "hail",
            "drizzle",
            "thunder",
        }

        def is_expletive_it(assertion, value: str) -> bool:
            return (
                value == "it"
                and assertion.predicate == "Evaluation"
                and assertion.relation is not None
                and assertion.relation.casefold() in expletive_it_relations
            )

        # Preserve the existing standalone-assertion behavior.
        repaired_assertions = []

        for assertion in repaired.assertions:
            filtered_arguments = []

            for argument in assertion.arguments:
                value = argument.value.casefold()

                if value not in pronouns:
                    filtered_arguments.append(argument)
                    continue

                if is_expletive_it(assertion, value):
                    filtered_arguments.append(argument)
                    continue

                # Otherwise the raw pronoun is unresolved and is removed.

            if len(filtered_arguments) == len(assertion.arguments):
                repaired_assertions.append(assertion)
                continue

            if not filtered_arguments:
                continue

            assertion.arguments = filtered_arguments

            if assertion.predicate != "Evaluation":
                schema = PREDICATE_SCHEMAS.get(assertion.predicate)

                if isinstance(schema, dict):
                    expected_arity = schema.get("arity")

                    if (
                        isinstance(expected_arity, int)
                        and len(assertion.arguments) != expected_arity
                    ):
                        continue

            repaired_assertions.append(assertion)

        # If standalone repair would remove every assertion and there are no rules,
        # fail before assigning an invalid empty state to the Pydantic model.
        if not repaired_assertions and not repaired.rules:
            raise SemanticParseError(
                "No grounded assertions remain after unresolved pronoun repair"
            )

        repaired.assertions = repaired_assertions

        # Repair pronouns inside rules.
        for rule in repaired.rules:
            generic_candidates = set()

            for assertion in rule.antecedents:
                for argument in assertion.arguments:
                    value = argument.value

                    if value.startswith("$") or value.casefold() in {
                        "someone",
                        "something",
                    }:
                        generic_candidates.add(value)

            replacement = (
                next(iter(generic_candidates)) if len(generic_candidates) == 1 else None
            )

            for assertion in [
                *rule.antecedents,
                *rule.consequents,
            ]:
                for argument in assertion.arguments:
                    value = argument.value.casefold()

                    if value not in pronouns:
                        continue

                    if is_expletive_it(assertion, value):
                        continue

                    if replacement is None:
                        raise SemanticParseError(
                            "Unresolved pronoun inside conditional rule: "
                            f"{argument.value!r}"
                        )

                    argument.value = replacement

        return repaired

    @staticmethod
    def validate_predicates(result: SemanticParseResult) -> SemanticParseResult:
        """Validate predicate and fallback consistency."""
        for assertion in _iter_all_assertions(result):
            predicate = assertion.predicate
            if predicate not in PREDICATE_SCHEMAS:
                raise SemanticParseError(f"Unknown semantic predicate: {predicate!r}")
            if predicate == "Evaluation":
                if not assertion.fallback:
                    raise SemanticParseError("Evaluation must have fallback=true")
                if not assertion.relation:
                    raise SemanticParseError("Evaluation requires a relation")
            else:
                if assertion.fallback:
                    raise SemanticParseError(
                        f"{predicate} is a known predicate, so fallback must be false"
                    )
                if assertion.relation is not None:
                    raise SemanticParseError(f"{predicate} should have relation=null")
        return result

    @staticmethod
    def validate_arguments(result: SemanticParseResult) -> SemanticParseResult:
        """Validate assertion argument counts against the YAML schema."""
        for assertion in _iter_all_assertions(result):
            schema = PREDICATE_SCHEMAS.get(assertion.predicate)
            if not isinstance(schema, dict):
                raise SemanticParseError(
                    f"No schema defined for predicate {assertion.predicate!r}"
                )
            actual_arity = len(assertion.arguments)
            if schema.get("variable_arity") is True:
                minimum, maximum = schema.get("min_arity"), schema.get("max_arity")
                if not isinstance(minimum, int) or minimum < 0:
                    raise ValueError(f"Invalid min_arity for {assertion.predicate!r}")
                if maximum is not None and (
                    not isinstance(maximum, int) or maximum < minimum
                ):
                    raise ValueError(f"Invalid max_arity for {assertion.predicate!r}")
                if actual_arity < minimum:
                    raise SemanticParseError(
                        f"{assertion.predicate} requires at least {minimum} arguments, "
                        f"got {actual_arity}"
                    )
                if isinstance(maximum, int) and actual_arity > maximum:
                    raise SemanticParseError(
                        f"{assertion.predicate} allows at most {maximum} arguments, "
                        f"got {actual_arity}"
                    )
                continue

            expected_arity = schema.get("arity")
            if not isinstance(expected_arity, int):
                raise ValueError(f"Invalid arity schema for {assertion.predicate!r}")
            if actual_arity != expected_arity:
                raise SemanticParseError(
                    f"{assertion.predicate} requires {expected_arity} arguments, "
                    f"got {actual_arity}"
                )
            expected_roles = schema.get("roles")
            actual_roles = [argument.role for argument in assertion.arguments]
            if actual_roles != expected_roles:
                raise SemanticParseError(
                    f"{assertion.predicate} requires argument roles "
                    f"{expected_roles!r}, got {actual_roles!r}"
                )
        return result

    @staticmethod
    def repair_rule_duplication(
        result: SemanticParseResult,
    ) -> SemanticParseResult:
        """Remove standalone assertions redundantly emitted with rules.

        Exact duplicates are always removed.

        Also remove a hypothetical standalone assertion when:
        - it contains a MeTTa variable;
        - that variable also occurs inside a rule; and
        - its source span is contained within that rule's source span.

        This handles redundant universal-rule output such as emitting both
        PropertyOf($x, rough) and the rule
        PropertyOf($x, smart) -> PropertyOf($x, rough),
        without attempting broader semantic rewriting.
        """
        if not result.rules or not result.assertions:
            return result

        repaired = result.model_copy(deep=True)

        rule_assertions = {
            assertion.model_dump_json()
            for rule in repaired.rules
            for assertion in [
                *rule.antecedents,
                *rule.consequents,
            ]
        }

        def assertion_variables(assertion) -> set[str]:
            return {
                argument.value
                for argument in assertion.arguments
                if argument.value.startswith("$")
            }

        rule_metadata = []

        for rule in repaired.rules:
            variables = {
                variable
                for assertion in [
                    *rule.antecedents,
                    *rule.consequents,
                ]
                for variable in assertion_variables(assertion)
            }

            rule_span = " ".join(rule.source_span.split()).casefold()

            rule_metadata.append((variables, rule_span))

        kept_assertions = []

        for assertion in repaired.assertions:
            if assertion.model_dump_json() in rule_assertions:
                continue

            assertion_vars = assertion_variables(assertion)

            assertion_span = " ".join(assertion.source_span.split()).casefold()

            redundant_rule_assertion = False

            if assertion.factuality == "hypothetical" and assertion_vars:
                for rule_vars, rule_span in rule_metadata:
                    if (
                        assertion_vars & rule_vars
                        and assertion_span
                        and assertion_span in rule_span
                    ):
                        redundant_rule_assertion = True
                        break

            if not redundant_rule_assertion:
                kept_assertions.append(assertion)

        repaired.assertions = kept_assertions

        return repaired

    @staticmethod
    def validate_source_spans(
        result: SemanticParseResult,
        sentence: str,
        context: str = "",
    ) -> SemanticParseResult:
        """Ensure every source span is grounded in the supplied text."""
        source_text = " ".join(
            part.strip() for part in (context, sentence) if part.strip()
        )

        if not source_text:
            raise SemanticParseError("Source text cannot be empty")

        source_text_normalized = " ".join(source_text.split()).casefold()

        for assertion in _iter_all_assertions(result):
            span = " ".join(assertion.source_span.split()).strip()

            if not span:
                raise SemanticParseError("source_span cannot be empty")

            if span.casefold() not in source_text_normalized:
                raise SemanticParseError(
                    f"source_span is not supported by the input text: {span!r}"
                )

        for rule in result.rules:
            span = " ".join(rule.source_span.split()).strip()

            if not span:
                raise SemanticParseError("rule source_span cannot be empty")

            if span.casefold() not in source_text_normalized:
                raise SemanticParseError(
                    f"rule source_span is not supported by the input text: {span!r}"
                )

        return result

    @staticmethod
    def render_metta(result: SemanticParseResult) -> list[str]:
        """Convert structured semantics into deterministic MeTTa expressions."""
        try:
            return render_semantic_result(result)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def validate_rendered_metta(expressions: list[str]) -> list[LinkAtom]:
        """Validate rendered MeTTa and convert it into LinkAtom objects."""
        try:
            return parse_rendered_metta(expressions)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def split_top_level_atoms(output: str) -> list[str]:
        """Split whitespace-separated top-level links without splitting nesting."""
        atoms: list[str] = []
        depth = 0
        start: int | None = None

        for idx, ch in enumerate(output):
            if ch.isspace() and depth == 0:
                continue
            if ch == "(":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise SemanticParseError("Unbalanced parentheses")
                if depth == 0 and start is not None:
                    atoms.append(output[start : idx + 1])
                    start = None
            elif depth == 0:
                raise SemanticParseError("Top-level LinkAtom expected")

        if depth != 0:
            raise SemanticParseError("Unbalanced parentheses")
        if not atoms:
            raise SemanticParseError("No MeTTa expressions")
        return atoms

    def _generate_text(self, prompt: str) -> str:
        """Generate text using either backend or saved model."""
        if self._model is not None and self._tokenizer is not None:
            # Distilled: use saved model
            enc = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            new_ids = out_ids[0][prompt_len:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        elif self._backend is not None:
            return self._backend.generate(prompt=prompt, model=self.model_name)

        else:
            raise ValueError("No backend or model available")

    def _generate_response(
        self, prompt: str, *, output_schema: dict[str, Any] | None = None
    ) -> str:
        """Execute a document-target request and clean only its response envelope."""
        generate_json = getattr(self._backend, "generate_json", None)
        if (
            output_schema is not None
            and self._require_structured_output
            and not callable(generate_json)
        ):
            raise ValueError(
                "The selected backend does not support JSON-schema decoding"
            )
        try:
            if output_schema is not None and callable(generate_json):
                output = generate_json(
                    prompt=prompt, model=self.model_name, schema=output_schema
                )
            else:
                output = self._generate_text(prompt)
        except Exception as error:
            raise ModelGenerationError(
                f"{self.parser_role} parser failed with model {self.model_name!r}: "
                f"backend raised {type(error).__name__}"
            ) from error
        return self.clean_model_output(output)

    def generate_document_target(
        self, source_text: str, *, catalog: OntologyCatalog | None = None
    ) -> tuple[dict[str, Any], str]:
        """Generate the quote-based document target with this parser's model.

        This is a draft, not an accepted document. Call parse_document() for
        source anchoring, contract validation, and ontology validation.
        """
        active_catalog = catalog if catalog is not None else self._catalog
        if active_catalog is None:
            active_catalog = load_ontology_catalog()
        prompt = build_catalog_document_prompt(source_text, catalog=active_catalog)
        from parser.semantic.catalog_output_schema import catalog_output_schema

        cleaned = self._generate_response(
            prompt.text, output_schema=catalog_output_schema(active_catalog)
        )
        if not cleaned:
            raise SemanticParseError("The model returned an empty response")
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise SemanticParseError("The model returned invalid JSON") from error
        if not isinstance(payload, dict) or set(payload) not in (
            {"entities", "assertions"},
            {"entities", "assertions", "rules"},
        ):
            raise SemanticParseError(
                "Document output must contain entities, assertions, and optional rules"
            )
        if not isinstance(payload["entities"], list) or not isinstance(
            payload["assertions"], list
        ):
            raise SemanticParseError("Catalog entities and assertions must be lists")
        if "rules" in payload and not isinstance(payload["rules"], list):
            raise SemanticParseError("Document rules must be a list")
        return payload, prompt.version

    def parse_document(
        self, *, document_id: str, corpus_id: str, source_text: str
    ) -> SemanticDocument:
        """Produce the miner-facing target without a second parser invocation."""
        document, _ = self.parse_document_with_target(
            document_id=document_id, corpus_id=corpus_id, source_text=source_text
        )
        return document

    def parse_document_with_target(
        self, *, document_id: str, corpus_id: str, source_text: str
    ) -> tuple[SemanticDocument, dict[str, object]]:
        """Validate one generation and retain its exact student label.

        The shared document library runs locally. Generating a compatibility
        view also requires that library, but neither operation starts PeTTa.
        """
        from parser.semantic.catalog_document_parser import (
            CATALOG_PARSER_VERSION,
            prepare_catalog_target,
        )

        catalog = (
            self._catalog if self._catalog is not None else load_ontology_catalog()
        )
        payload, prompt_version = self.generate_document_target(
            source_text, catalog=catalog
        )
        document, payload = prepare_catalog_target(
            payload,
            document_id=document_id,
            corpus_id=corpus_id,
            source_text=source_text,
            parser_provenance={
                "model": self.model_name,
                "parser_version": CATALOG_PARSER_VERSION,
                "prompt_version": prompt_version,
                "ontology_versions": catalog.versions,
                "provider": self.provider_name,
                "model_revision": self._model_revision,
            },
            catalog=catalog,
        )
        return document, payload

    @classmethod
    def read_legacy_result(
        cls,
        output: str,
        *,
        sentence: str,
        context: str = "",
        aliases: Mapping[str, str] | None = None,
        alias_types: Mapping[str, str] | None = None,
    ) -> SemanticParseResult:
        """Read historical records without invoking a teacher or student."""
        if not sentence.strip():
            raise ValueError("The sentence cannot be empty")
        cleaned = cls.clean_model_output(output)
        if not cleaned:
            raise SemanticParseError("The legacy record is empty")
        try:
            result = SemanticParseResult.model_validate_json(cleaned)
        except ValidationError as error:
            raise SemanticParseError(
                f"Invalid structured legacy record: {error}"
            ) from error
        try:
            result = normalize_semantic_result(
                result, aliases=aliases, alias_types=alias_types
            )
        except SemanticNormalizationError as error:
            raise SemanticParseError(str(error)) from error
        result = cls.repair_ambiguous_pronouns(result)
        result = cls.repair_known_predicate_contract(result)
        result = cls.validate_predicates(result)
        result = cls.validate_arguments(result)
        result = cls.repair_rule_duplication(result)
        return cls.validate_source_spans(result, sentence=sentence, context=context)

    def generate_structured(
        self,
        sentence: str,
        context: str = "",
        *,
        aliases: Mapping[str, str] | None = None,
        alias_types: Mapping[str, str] | None = None,
    ) -> SemanticParseResult:
        """Derive a compatibility view from one canonical generation."""
        from parser.semantic.compatibility import document_to_legacy_result

        document = self.parse(sentence, context=context)
        return document_to_legacy_result(
            document, aliases=aliases, alias_types=alias_types
        )

    def parse(self, sentence: str, context: str = "") -> SemanticDocument:
        """Extract one document from context plus sentence, for either model.

        Context is part of the exact source and may itself yield assertions;
        it is not reference-only background for a target-sentence projection.
        """
        from hashlib import sha256

        if not sentence.strip():
            raise ValueError("The sentence cannot be empty")
        source_text = context + "\n" + sentence if context else sentence
        return self.parse_document(
            document_id="text-" + sha256(source_text.encode("utf-8")).hexdigest(),
            corpus_id="interactive",
            source_text=source_text,
        )


# ── Reference Parser (Teacher) ───────────────────────────────────────────────


class ReferenceSemanticParser(_BaseSemanticParser):
    """High-accuracy parser used for offline annotation and dataset generation."""

    parser_role = "reference"

    def __init__(
        self,
        *,
        backend: ModelBackend,
        config: SemanticParserConfig,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
        require_structured_output: bool = False,
    ):
        super().__init__(
            backend=backend,
            config=config,
            catalog=catalog,
            model_revision=model_revision,
            require_structured_output=require_structured_output,
        )


# ── Distilled Parser (Student) ──────────────────────────────────────────────


class DistilledSemanticParser(_BaseSemanticParser):
    """Local parser that loads a saved fine-tuned student model."""

    parser_role = "distilled"

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 3072,
        num_beams: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {"base_model": "unknown"}
        if not isinstance(self.metadata, dict):
            raise ValueError("Student training metadata must be a JSON object")
        from parser.semantic.catalog_student import validate_catalog_student_checkpoint

        catalog = load_ontology_catalog()
        validate_catalog_student_checkpoint(self.model_dir, catalog=catalog)

        # Load tokenizer from disk
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Preserve checkpoint precision only on BF16-capable CUDA devices.
        # An explicit CPU/MPS request must not depend on a GPU also being present.
        supports_bf16 = False
        if torch.cuda.is_available() and (
            device == "auto" or device.startswith("cuda")
        ):
            devices = (
                list(range(torch.cuda.device_count()))
                if device == "auto"
                else [torch.device(device).index]
            )
            supports_bf16 = bool(devices)
            for index in devices:
                with torch.cuda.device(index):
                    supports_bf16 = supports_bf16 and torch.cuda.is_bf16_supported()
        if not supports_bf16:
            print("Using float32 for distilled inference on the requested device")

        # Load model from disk
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype="auto" if supports_bf16 else torch.float32,
            device_map=device,
        )
        self.model.eval()

        # Create config
        config = SemanticParserConfig(
            model_name=self.metadata.get("base_model", "unknown"),
            prompt_version=self.metadata.get("prompt_version") or "2.0.0",
        )

        # Initialize base class with model and tokenizer
        super().__init__(
            backend=None,
            config=config,
            model=self.model,
            tokenizer=self.tokenizer,
            catalog=catalog,
        )

        print(f"DistilledSemanticParser loaded from {model_dir}")
        print(f"  Base model: {self.metadata.get('base_model', 'unknown')}")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 3072,
        num_beams: int = 4,
    ) -> DistilledSemanticParser:
        """Load a saved model for inference."""
        return cls(model_dir, device, max_new_tokens, num_beams)


# ── Builder functions ─────────────────────────────────────────────────────────


def build_distilled_parser(
    model_dir: str = "models/semantic_document_parser",
    device: str = "auto",
    max_new_tokens: int = 3072,
    num_beams: int = 4,
) -> DistilledSemanticParser:
    """Build a distilled parser from a saved model directory."""
    return DistilledSemanticParser.from_pretrained(
        model_dir=model_dir,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
