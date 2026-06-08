"""Build distilled feature training datasets from kani routing logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol, TypedDict

import httpx

from kani.classification_context import build_classification_input
from kani.config import load_config
from kani.dirs import data_dir, log_dir
from kani.scorer import SEMANTIC_DIMENSIONS

VALID_DIMENSION_LABELS = {"low", "medium", "high"}


class FeatureAnnotator(Protocol):
    def annotate(self, prompt: str) -> dict[str, str] | None: ...


class DistilledFeatureExample(TypedDict):
    prompt: str
    tokenCount: int
    codePresence: str
    reasoningMarkers: str
    technicalTerms: str
    creativeMarkers: str
    simpleIndicators: str
    multiStepPatterns: str
    questionComplexity: str
    imperativeVerbs: str
    constraintCount: str
    outputFormat: str
    referenceComplexity: str
    negationComplexity: str
    domainSpecificity: str
    agenticTask: str
    timestamp: str | None
    source: str


class LLMFeatureAnnotator:
    """Offline annotator that labels semantic dimensions with an LLM."""

    _PROMPT_TEMPLATE = (
        "You are labeling prompts for routing distillation. "
        "Return JSON object only with exactly these keys: "
        f"{', '.join(SEMANTIC_DIMENSIONS)}. "
        "Each value MUST be one of: low, medium, high. "
        "Do NOT use any other words or descriptions. "
        "If unsure, use 'medium'.\n\n"
        "Dimension definitions:\n"
        "- codePresence: Does the prompt contain or request code? "
        "low = no code whatsoever; medium = mentions code concepts; "
        "high = large code blocks, debugging, or code generation tasks.\n"
        "- reasoningMarkers: Does it ask for logical reasoning or chain-of-thought? "
        "low = factual lookup or simple Q&A; medium = some analysis needed; "
        "high = proofs, theorems, step-by-step deduction, multi-hop reasoning.\n"
        "- technicalTerms: How domain-specific is the vocabulary? "
        "low = everyday language; medium = some technical jargon; "
        "high = dense specialized terminology.\n"
        "- creativeMarkers: Is this a creative writing task? "
        "low = not creative; medium = lightly creative; "
        "high = story, poem, brainstorm, or open-ended creative generation.\n"
        "- simpleIndicators: How trivial is the request? "
        "low = complex or substantial (simpleIndicators is low for hard prompts); "
        "medium = moderately substantial; high = extremely simple "
        "('hello', 'what is X', 'define Y').\n"
        "- multiStepPatterns: Does it require multiple sequential steps? "
        "low = single step; medium = 2-3 steps; "
        "high = many ordered steps, numbered phases, or complex workflow.\n"
        "- questionComplexity: How deep or multi-faceted are the questions? "
        "low = single straightforward question; medium = 2-3 questions; "
        "high = deeply probing, multi-layered questioning.\n"
        "- imperativeVerbs: How action-oriented is the prompt? "
        "low = informational or descriptive; medium = some directives; "
        "high = dominated by build/implement/deploy/review commands.\n"
        "- constraintCount: How many explicit constraints or requirements? "
        "low = open-ended; medium = 1-2 conditions; "
        "high = many must/ensure/require/within/except constraints.\n"
        "- outputFormat: Is a specific output format demanded? "
        "low = no format specified; medium = vague hint; "
        "high = explicit json/csv/markdown/table specification.\n"
        "- referenceComplexity: Does it reference external or prior context? "
        "low = self-contained; medium = references prior conversation; "
        "high = complex multi-document or external-source references.\n"
        "- negationComplexity: How many negative constraints? "
        "low = positive framing; medium = some negation; "
        "high = many not/without/except/don't conditions.\n"
        "- domainSpecificity: Is this in a specialised domain? "
        "low = general knowledge; medium = somewhat specialised; "
        "high = deeply domain-specific (medical, legal, financial, scientific).\n"
        "- agenticTask: Does this involve autonomous tool use or file ops? "
        "low = pure informational query; medium = some tool interaction; "
        "high = read file/edit/execute/deploy/debug — agent-style action.\n\n"
        "Prompt:\n{prompt}"
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        cfg = None
        resolved = None
        try:
            loaded = load_config()
            cfg = loaded.feature_annotator
            resolved = loaded.feature_annotator_resolved()
        except Exception:
            pass

        self.model = (
            model
            or os.environ.get("KANI_LLM_ANNOTATOR_MODEL")
            or (cfg.model if cfg else None)
            or "google/gemini-2.5-flash-lite"
        )
        self.base_url = (
            base_url
            or os.environ.get("KANI_LLM_ANNOTATOR_BASE_URL")
            or (resolved[0] if resolved else None)
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("KANI_LLM_ANNOTATOR_API_KEY")
            or (resolved[1] if resolved else None)
            or os.environ.get("OPENROUTER_API_KEY", "")
        )

    def annotate(self, prompt: str) -> dict[str, str] | None:
        if not self.api_key:
            raise RuntimeError(
                "KANI_LLM_ANNOTATOR_API_KEY or OPENROUTER_API_KEY is required"
            )

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": self._PROMPT_TEMPLATE.format(
                                prompt=prompt[:3500]
                            ),
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        if not content:
            return None
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None

        labels: dict[str, str] = {}
        for dim in SEMANTIC_DIMENSIONS:
            value = str(parsed.get(dim, "")).strip().lower()
            if value not in VALID_DIMENSION_LABELS:
                return None
            labels[dim] = value
        return labels


def deterministic_token_count(prompt: str) -> int:
    return max(1, len(prompt.split()))


def load_routing_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def _validate_semantic_labels(labels: dict[str, str]) -> bool:
    for dim in SEMANTIC_DIMENSIONS:
        value = labels.get(dim)
        if value not in VALID_DIMENSION_LABELS:
            return False
    return True


def _extract_semantic_labels_from_record(
    record: dict[str, Any],
) -> dict[str, str] | None:
    signals = record.get("signals")
    if not isinstance(signals, dict):
        return None

    raw_labels = signals.get("semanticLabels")
    if not isinstance(raw_labels, dict):
        return None

    labels: dict[str, str] = {
        key: str(value).strip().lower() for key, value in raw_labels.items()
    }
    if not _validate_semantic_labels(labels):
        return None
    return {dim: labels[dim] for dim in SEMANTIC_DIMENSIONS}


def _classification_prompt_from_record(record: dict[str, Any]) -> str:
    context = record.get("classification_context")
    if isinstance(context, dict):
        context_text = str(context.get("text") or "").strip()
        if context_text:
            return context_text

    messages = record.get("messages")
    if isinstance(messages, list):
        try:
            return build_classification_input(messages).text
        except Exception:
            pass

    return str(record.get("prompt") or record.get("prompt_preview") or "").strip()


def _make_example(
    prompt: str,
    labels: dict[str, str],
    record: dict[str, Any],
    source: str,
) -> DistilledFeatureExample:
    return {
        "prompt": prompt,
        "tokenCount": deterministic_token_count(prompt),
        "codePresence": labels["codePresence"],
        "reasoningMarkers": labels["reasoningMarkers"],
        "technicalTerms": labels["technicalTerms"],
        "creativeMarkers": labels["creativeMarkers"],
        "simpleIndicators": labels["simpleIndicators"],
        "multiStepPatterns": labels["multiStepPatterns"],
        "questionComplexity": labels["questionComplexity"],
        "imperativeVerbs": labels["imperativeVerbs"],
        "constraintCount": labels["constraintCount"],
        "outputFormat": labels["outputFormat"],
        "referenceComplexity": labels["referenceComplexity"],
        "negationComplexity": labels["negationComplexity"],
        "domainSpecificity": labels["domainSpecificity"],
        "agenticTask": labels["agenticTask"],
        "timestamp": str(record.get("timestamp")) if record.get("timestamp") else None,
        "source": source,
    }


def _save_examples(
    latest_by_prompt: dict[str, DistilledFeatureExample],
    output_path: Path,
) -> list[DistilledFeatureExample]:
    examples = sorted(
        latest_by_prompt.values(),
        key=lambda item: ((item["timestamp"] or ""), item["prompt"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return examples


_CHECKPOINT_INTERVAL = 1


def extract_distilled_feature_examples(
    records: list[dict[str, Any]],
    *,
    annotator: FeatureAnnotator | None = None,
    checkpoint_path: Path | None = None,
) -> list[DistilledFeatureExample]:
    latest_by_prompt: dict[str, DistilledFeatureExample] = {}

    if checkpoint_path and checkpoint_path.exists():
        try:
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    prompt = item.get("prompt", "")
                    if prompt:
                        latest_by_prompt[prompt] = item
                print(f"  Resumed {len(latest_by_prompt)} examples from checkpoint")
        except (json.JSONDecodeError, OSError):
            pass

    annotated_since_save = 0
    skipped = 0
    total = len(records)

    for idx, record in enumerate(records, 1):
        prompt = _classification_prompt_from_record(record)
        if not prompt:
            skipped += 1
            print(f"  [{idx}/{total}] skip: empty prompt")
            continue

        labels = _extract_semantic_labels_from_record(record)
        source = "log"
        if labels is None and annotator is not None:
            if prompt in latest_by_prompt:
                skipped += 1
                print(f"  [{idx}/{total}] skip: duplicate")
                continue
            print(f"  [{idx}/{total}] annotate: {prompt[:120].replace(chr(10), ' ')}")
            labels = annotator.annotate(prompt)
            source = "annotated"
            annotated_since_save += 1

        if labels is None:
            skipped += 1
            print(f"  [{idx}/{total}] skip: no labels returned")
            continue

        if not _validate_semantic_labels(labels):
            continue

        example = _make_example(prompt, labels, record, source)

        current = latest_by_prompt.get(prompt)
        if current is None or (example["timestamp"] or "") >= (
            current["timestamp"] or ""
        ):
            latest_by_prompt[prompt] = example

        if checkpoint_path and annotated_since_save >= _CHECKPOINT_INTERVAL:
            _save_examples(latest_by_prompt, checkpoint_path)
            print(
                f"  [{idx}/{total}] checkpoint: {len(latest_by_prompt)} examples saved"
            )
            annotated_since_save = 0

    return sorted(
        latest_by_prompt.values(),
        key=lambda item: ((item["timestamp"] or ""), item["prompt"]),
    )


def build_feature_dataset(
    log_paths: list[Path],
    output_path: Path,
    *,
    annotator: FeatureAnnotator | None = None,
) -> list[DistilledFeatureExample]:
    examples = extract_distilled_feature_examples(
        load_routing_records(log_paths),
        annotator=annotator,
        checkpoint_path=output_path if annotator else None,
    )
    _save_examples(
        {e["prompt"]: e for e in examples},
        output_path,
    )
    return examples


def resolve_log_paths(
    paths: list[str], *, log_directory: Path, pattern: str
) -> list[Path]:
    if paths:
        return [Path(path).expanduser() for path in paths]
    return sorted(log_directory.expanduser().glob(pattern))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build distilled semantic feature dataset from routing logs"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional routing log files. If omitted, scan --log-dir with --glob.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(log_dir()),
        help="Directory containing routing-*.jsonl files",
    )
    parser.add_argument(
        "--glob",
        default="routing-*.jsonl",
        help="Glob used when explicit paths are omitted",
    )
    parser.add_argument(
        "--output",
        default=str(data_dir() / "distilled_feature_dataset.json"),
        help="Output JSON dataset path",
    )
    parser.add_argument(
        "--annotate-missing",
        action="store_true",
        help="Use LLM annotation for records missing semantic labels",
    )
    parser.add_argument("--model", help="LLM model for annotation")
    parser.add_argument("--base-url", help="LLM base URL for annotation")
    parser.add_argument("--api-key", help="LLM API key for annotation")
    args = parser.parse_args(argv)

    log_paths = resolve_log_paths(
        args.paths,
        log_directory=Path(args.log_dir),
        pattern=args.glob,
    )
    if not log_paths:
        parser.error("No routing log files found")

    annotator = None
    if args.annotate_missing:
        annotator = LLMFeatureAnnotator(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )

    output_path = Path(args.output).expanduser()
    examples = build_feature_dataset(log_paths, output_path, annotator=annotator)

    print(f"Loaded {len(log_paths)} log files")
    print(f"Wrote {len(examples)} distilled feature examples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
