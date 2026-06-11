"""Kani scoring engine.

Distilled feature-based prompt classifier used by the router.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

log = logging.getLogger(__name__)


FEATURE_DIMENSIONS: tuple[str, ...] = (
    "tokenCount",
    "codePresence",
    "reasoningMarkers",
    "technicalTerms",
    "creativeMarkers",
    "simpleIndicators",
    "multiStepPatterns",
    "questionComplexity",
    "imperativeVerbs",
    "constraintCount",
    "outputFormat",
    "referenceComplexity",
    "negationComplexity",
    "domainSpecificity",
    "agenticTask",
)

SEMANTIC_DIMENSIONS: tuple[str, ...] = FEATURE_DIMENSIONS[1:]

_DIMENSION_VALUE_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}


class Tier(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


class ScoringConfig(BaseModel):
    """Configuration for the distilled feature scoring pipeline."""

    disable_axis_overrides: bool = False
    fallback_tier: Tier = Tier.MEDIUM
    fallback_confidence: float = 0.35


@dataclass
class DimensionResult:
    name: str
    raw_score: float
    weight: float
    weighted_score: float
    match_count: int = 0


@dataclass
class ClassificationResult:
    score: float
    tier: Tier
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)
    agentic_score: float = 0.0
    dimensions: list[DimensionResult] = field(default_factory=list)


_DEFAULT_WEIGHTS: dict[str, float] = {
    "tokenCount": 0.15,
    "codePresence": 1.0,
    "reasoningMarkers": 1.4,
    "technicalTerms": 1.1,
    "creativeMarkers": 0.8,
    "simpleIndicators": 1.0,
    "multiStepPatterns": 1.3,
    "questionComplexity": 1.2,
    "imperativeVerbs": 0.9,
    "constraintCount": 1.2,
    "outputFormat": 0.9,
    "referenceComplexity": 1.1,
    "negationComplexity": 0.9,
    "domainSpecificity": 1.1,
    "agenticTask": 1.4,
}

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "SIMPLE": 0.2,
    "MEDIUM": 0.55,
    "COMPLEX": 0.75,
}


def _build_embedding_client(
    model_name: str = "text-embedding-3-small",
) -> tuple[Any, str]:
    """Create an embedding client from config or environment variables."""
    from openai import OpenAI

    from kani.config import load_config

    try:
        loaded = load_config()
        cfg = loaded.embedding
    except Exception:
        loaded = None
        cfg = None

    if loaded and cfg and cfg.enabled:
        resolved = loaded.embedding_resolved()
        if resolved is not None:
            base_url, api_key = resolved
            return (
                OpenAI(api_key=api_key or "dummy", base_url=base_url),
                cfg.model or model_name,
            )

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    base_url = None
    resolved_model = model_name

    if not os.environ.get("OPENAI_API_KEY") and os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        base_url = "https://openrouter.ai/api/v1"
        if not resolved_model.startswith("openai/"):
            resolved_model = f"openai/{resolved_model}"

    if not api_key:
        raise RuntimeError(
            "No embedding config or OPENAI_API_KEY / OPENROUTER_API_KEY"
        )

    return OpenAI(api_key=api_key, base_url=base_url), resolved_model


_CODE_MARKERS = (
    "```",
    "traceback",
    "exception",
    "error",
    "def ",
    "class ",
    "function",
    "import ",
    "npm ",
    "uv ",
    "docker ",
)
_REASONING_MARKERS = (
    "why",
    "explain",
    "prove",
    "reason",
    "analyze",
    "compare",
    "root cause",
    "なぜ",
    "理由",
    "原因",
    "証明",
    "分析",
)
_TECHNICAL_TERMS = (
    "api",
    "http",
    "json",
    "yaml",
    "docker",
    "pytest",
    "pyright",
    "fastapi",
    "pydantic",
    "embedding",
    "classifier",
    "routing",
    "provider",
    "config",
)
_CREATIVE_MARKERS = (
    "write",
    "draft",
    "copy",
    "design",
    "story",
    "poem",
    "文章",
    "デザイン",
)
_SIMPLE_INDICATORS = ("hi", "hello", "thanks", "ok", "yes", "no", "はい", "ありがとう")
_MULTI_STEP_MARKERS = (
    "then",
    "after",
    "first",
    "second",
    "複数",
    "手順",
    "まず",
    "次に",
)
_FORMAT_MARKERS = ("json", "yaml", "markdown", "table", "list", "箇条書き", "表")
_NEGATION_MARKERS = ("not", "never", "without", "don't", "ない", "禁止", "不要")
_AGENTIC_MARKERS = (
    "fix",
    "implement",
    "run",
    "test",
    "debug",
    "investigate",
    "修正",
    "実装",
    "確認",
)
_IMPERATIVE_MARKERS = (
    "fix",
    "add",
    "remove",
    "update",
    "run",
    "check",
    "修正",
    "追加",
    "更新",
    "確認",
)


def _label_from_count(count: int, *, medium: int = 1, high: int = 3) -> str:
    if count >= high:
        return "high"
    if count >= medium:
        return "medium"
    return "low"


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker in lowered)


def _heuristic_semantic_labels(text: str) -> tuple[dict[str, str], float]:
    token_count = _token_count(text)
    question_count = text.count("?") + text.count("？")
    line_count = max(1, text.count("\n") + 1)
    code_count = _count_markers(text, _CODE_MARKERS)
    technical_count = _count_markers(text, _TECHNICAL_TERMS)
    reasoning_count = _count_markers(text, _REASONING_MARKERS)
    imperative_count = _count_markers(text, _IMPERATIVE_MARKERS)
    multi_step_count = _count_markers(text, _MULTI_STEP_MARKERS)
    format_count = _count_markers(text, _FORMAT_MARKERS)
    negation_count = _count_markers(text, _NEGATION_MARKERS)
    agentic_count = _count_markers(text, _AGENTIC_MARKERS)

    labels = {
        "codePresence": _label_from_count(code_count, high=2),
        "reasoningMarkers": _label_from_count(reasoning_count, high=2),
        "technicalTerms": _label_from_count(technical_count, high=3),
        "creativeMarkers": _label_from_count(
            _count_markers(text, _CREATIVE_MARKERS), high=2
        ),
        "simpleIndicators": "high"
        if token_count <= 4 and _count_markers(text, _SIMPLE_INDICATORS)
        else "low",
        "multiStepPatterns": _label_from_count(
            multi_step_count + int(line_count >= 4), high=3
        ),
        "questionComplexity": _label_from_count(
            question_count + int(token_count >= 80), high=2
        ),
        "imperativeVerbs": _label_from_count(imperative_count, high=2),
        "constraintCount": _label_from_count(
            text.count("must") + text.count("必須") + negation_count, high=3
        ),
        "outputFormat": _label_from_count(format_count, high=2),
        "referenceComplexity": _label_from_count(
            text.count("http") + text.count("file:") + text.count("src/"), high=2
        ),
        "negationComplexity": _label_from_count(negation_count, high=2),
        "domainSpecificity": _label_from_count(technical_count + code_count, high=4),
        "agenticTask": _label_from_count(agentic_count, high=2),
    }
    confidence = 0.85 if any(value != "low" for value in labels.values()) else 0.65
    return labels, confidence


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _tier_from_score(score: float, thresholds: dict[str, float]) -> Tier:
    simple_max = float(thresholds.get("SIMPLE", _DEFAULT_THRESHOLDS["SIMPLE"]))
    medium_max = float(thresholds.get("MEDIUM", _DEFAULT_THRESHOLDS["MEDIUM"]))
    complex_max = float(thresholds.get("COMPLEX", _DEFAULT_THRESHOLDS["COMPLEX"]))

    if score <= simple_max:
        return Tier.SIMPLE
    if score <= medium_max:
        return Tier.MEDIUM
    if score <= complex_max:
        return Tier.COMPLEX
    return Tier.REASONING


def _semantic_axis_score(
    semantic_labels: dict[str, str],
    names: list[str],
) -> float:
    values = [
        _DIMENSION_VALUE_MAP.get(semantic_labels.get(name, "low"), 0.0)
        for name in names
    ]
    return sum(values) / max(len(values), 1)


def _tier_from_axes(
    score: float,
    semantic_labels: dict[str, str],
    thresholds: dict[str, float],
    *,
    disable_axis_overrides: bool = False,
) -> Tier:
    """Compute final tier from base score and semantic dimension labels.

    Args:
        score: Weighted composite score from all dimensions.
        semantic_labels: Dict of dimension name -> ``'low'`` | ``'medium'`` | ``'high'``.
        thresholds: Dict of tier name -> score threshold.
        disable_axis_overrides: If True, skip axis-based promotions (agenticTask,
            reasoningMarkers, complexity_score) and return ``base_tier`` directly.

    Returns:
        Final tier.
    """
    base_tier = _tier_from_score(score, thresholds)
    if disable_axis_overrides:
        return base_tier

    complexity_score = _semantic_axis_score(
        semantic_labels,
        [
            "codePresence",
            "multiStepPatterns",
            "constraintCount",
            "imperativeVerbs",
            "domainSpecificity",
            "technicalTerms",
        ],
    )
    reasoning_score = _semantic_axis_score(
        semantic_labels,
        [
            "reasoningMarkers",
            "questionComplexity",
            "referenceComplexity",
            "negationComplexity",
        ],
    )

    axis_tier = Tier.SIMPLE
    if semantic_labels.get("agenticTask") == "high" and (
        reasoning_score >= 0.75
        or semantic_labels.get("reasoningMarkers") == "high"
    ):
        axis_tier = Tier.REASONING
    elif (
        semantic_labels.get("agenticTask") == "high"
        and semantic_labels.get("imperativeVerbs") == "high"
    ):
        axis_tier = Tier.MEDIUM
    elif complexity_score >= 0.8:
        axis_tier = Tier.COMPLEX
    elif complexity_score >= 0.5:
        axis_tier = Tier.MEDIUM

    return max(base_tier, axis_tier, key=lambda tier: list(Tier).index(tier))


class DistilledFeatureClassifier:
    """Embedding-based multi-output semantic feature classifier loaded from pkl."""

    _instance: DistilledFeatureClassifier | None = None
    _model_dir: Path | None = None

    def __init__(self, model_path: Path) -> None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.classifier = data["classifier"]
        self.embedding_model: str = data.get(
            "embedding_model", "text-embedding-3-small"
        )
        self.semantic_dimensions: list[str] = list(
            data.get("semantic_dimensions", SEMANTIC_DIMENSIONS)
        )
        self.label_encoders: dict[str, Any] = data["label_encoders"]
        self.weights: dict[str, float] = dict(data.get("weights", {}))
        self.tier_thresholds: dict[str, float] = dict(
            data.get(
                "tier_thresholds",
                _DEFAULT_THRESHOLDS,
            )
        )
        self._client: Any | None = None

    @classmethod
    def load(
        cls, model_dir: Path | None = None
    ) -> DistilledFeatureClassifier | None:
        if model_dir is None:
            model_dir = (
                Path(__file__).resolve().parent.parent.parent / "models"
            )
        if cls._instance is not None and cls._model_dir == model_dir:
            return cls._instance
        pkl = model_dir / "feature_classifier.pkl"
        if not pkl.exists():
            return None
        try:
            cls._instance = cls(pkl)
            cls._model_dir = model_dir
            log.info("Loaded distilled feature classifier from %s", pkl)
            return cls._instance
        except Exception:
            log.exception("Failed to load distilled feature classifier")
            return None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client, self.embedding_model = _build_embedding_client(
            self.embedding_model
        )
        return self._client

    def _embed(self, text: str) -> np.ndarray:
        client = self._get_client()
        resp = client.embeddings.create(
            input=[text[:4000]], model=self.embedding_model
        )
        return np.array([resp.data[0].embedding], dtype=np.float32)

    def predict(self, text: str) -> tuple[dict[str, str], float]:
        embedding = self._embed(text)
        predicted = self.classifier.predict(embedding)[0]
        probs = self.classifier.predict_proba(embedding)

        labels: dict[str, str] = {}
        confidence_sum = 0.0
        for idx, dim in enumerate(self.semantic_dimensions):
            if dim in self.label_encoders:
                encoder = self.label_encoders[dim]
                label = encoder.inverse_transform([predicted[idx]])[0]
                labels[dim] = str(label)
                dim_probs = dict(
                    zip(
                        [str(c) for c in encoder.classes_],
                        [float(p) for p in probs[idx][0]],
                    )
                )
                confidence_sum += max(dim_probs.values(), default=0.0)

        confidence = confidence_sum / max(len(self.semantic_dimensions), 1)
        return labels, confidence


class Scorer:
    """Distilled feature-based prompt classifier."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        *,
        feature_model_dir: Any | None = None,
        enable_routing_log: bool = True,
    ) -> None:
        self.config = config or ScoringConfig()
        self._enable_routing_log = enable_routing_log
        self._feature_clf = DistilledFeatureClassifier.load(feature_model_dir)
        self._pkl_cooldown_until: float = 0.0

    @staticmethod
    def _build_dimensions(
        token_count: int,
        semantic_labels: dict[str, str],
        weights: dict[str, float],
    ) -> tuple[list[DimensionResult], float, float]:
        dimensions: list[DimensionResult] = []

        token_value = min(token_count / 2000.0, 1.0)
        token_weight = float(weights.get("tokenCount", 0.15))
        dimensions.append(
            DimensionResult(
                name="tokenCount",
                raw_score=token_value,
                weight=token_weight,
                weighted_score=token_value * token_weight,
            )
        )

        total_weighted = token_value * token_weight
        total_weight = token_weight
        agentic_score = 0.0

        for dim in SEMANTIC_DIMENSIONS:
            label = semantic_labels.get(dim, "low")
            value = _DIMENSION_VALUE_MAP.get(label, 0.0)
            weight = float(weights.get(dim, 1.0))
            weighted = value * weight
            dimensions.append(
                DimensionResult(
                    name=dim,
                    raw_score=value,
                    weight=weight,
                    weighted_score=weighted,
                )
            )
            if dim == "agenticTask":
                agentic_score = value
            total_weighted += weighted
            total_weight += weight

        score = total_weighted / max(total_weight, 1e-9)
        return dimensions, score, agentic_score

    def _classify_with_features(self, text: str) -> ClassificationResult:
        token_count = _token_count(text)

        _PKL_RETRY_SECONDS: float = 60.0

        if self._feature_clf is not None and time.time() >= self._pkl_cooldown_until:
            try:
                semantic_labels, confidence = self._feature_clf.predict(text)
                weights = self._feature_clf.weights or _DEFAULT_WEIGHTS
                thresholds = self._feature_clf.tier_thresholds
                method_signals = {
                    "raw": "distilled-features",
                    "matches": 0,
                }
            except Exception:
                log.warning(
                    "Feature classifier predict failed, "
                    "using heuristics for %ds",
                    int(_PKL_RETRY_SECONDS),
                )
                self._pkl_cooldown_until = time.time() + _PKL_RETRY_SECONDS
                semantic_labels, confidence = _heuristic_semantic_labels(text)
                weights = _DEFAULT_WEIGHTS
                thresholds = _DEFAULT_THRESHOLDS
                method_signals = {
                    "raw": "heuristic-features",
                    "matches": 0,
                }
        else:
            semantic_labels, confidence = _heuristic_semantic_labels(text)
            weights = _DEFAULT_WEIGHTS
            thresholds = _DEFAULT_THRESHOLDS
            method_signals = {
                "raw": "heuristic-features",
                "matches": 0,
            }

        dimensions, score, agentic_score = self._build_dimensions(
            token_count,
            semantic_labels,
            weights,
        )
        tier = _tier_from_axes(
            score, semantic_labels, thresholds,
            disable_axis_overrides=self.config.disable_axis_overrides,
        )

        signals: dict[str, Any] = {
            "method": method_signals,
            "tokenCount": token_count,
            "semanticLabels": semantic_labels,
            "featureVersion": "v1",
        }

        result = ClassificationResult(
            score=score,
            tier=tier,
            confidence=confidence,
            signals=signals,
            agentic_score=agentic_score,
            dimensions=dimensions,
        )

        if self._enable_routing_log:
            from kani.logger import RoutingLogger

            RoutingLogger.log(text, result)

        return result

    def classify(self, text: str) -> ClassificationResult:
        log.debug("Scoring classification input text_length=%d", len(text))
        try:
            return self._classify_with_features(text)
        except Exception:
            log.exception("Feature classification failed, falling back")

        result = ClassificationResult(
            score=0.0,
            tier=self.config.fallback_tier,
            confidence=self.config.fallback_confidence,
            signals={"method": {"raw": "default", "matches": 0}},
            agentic_score=0.0,
            dimensions=[],
        )
        if self._enable_routing_log:
            from kani.logger import RoutingLogger

            RoutingLogger.log(text, result)
        return result
