"""Tests for kani distilled feature scoring engine."""

from __future__ import annotations

from unittest.mock import patch

from kani.scorer import ClassificationResult, Scorer, ScoringConfig, Tier


class _StubFeatureClassifier:
    def __init__(self) -> None:
        self.weights = {
            "tokenCount": 0.2,
            "codePresence": 1.0,
            "reasoningMarkers": 1.3,
            "technicalTerms": 1.0,
            "creativeMarkers": 0.8,
            "simpleIndicators": 1.0,
            "multiStepPatterns": 1.2,
            "questionComplexity": 1.1,
            "imperativeVerbs": 0.9,
            "constraintCount": 1.0,
            "outputFormat": 0.8,
            "referenceComplexity": 0.9,
            "negationComplexity": 0.9,
            "domainSpecificity": 1.0,
            "agenticTask": 1.4,
        }
        self.tier_thresholds = {"SIMPLE": 0.2, "MEDIUM": 0.45, "COMPLEX": 0.7}

    def predict(self, text: str) -> tuple[dict[str, str], float]:
        if "prove" in text.lower():
            return (
                {
                    "codePresence": "medium",
                    "reasoningMarkers": "high",
                    "technicalTerms": "high",
                    "creativeMarkers": "low",
                    "simpleIndicators": "low",
                    "multiStepPatterns": "high",
                    "questionComplexity": "high",
                    "imperativeVerbs": "medium",
                    "constraintCount": "high",
                    "outputFormat": "medium",
                    "referenceComplexity": "high",
                    "negationComplexity": "medium",
                    "domainSpecificity": "high",
                    "agenticTask": "high",
                },
                0.88,
            )

        return (
            {
                "codePresence": "low",
                "reasoningMarkers": "low",
                "technicalTerms": "low",
                "creativeMarkers": "low",
                "simpleIndicators": "high",
                "multiStepPatterns": "low",
                "questionComplexity": "low",
                "imperativeVerbs": "low",
                "constraintCount": "low",
                "outputFormat": "low",
                "referenceComplexity": "low",
                "negationComplexity": "low",
                "domainSpecificity": "low",
                "agenticTask": "medium",
            },
            0.81,
        )


class TestDistilledFeatureScorer:
    def test_runtime_classifier_uses_no_embedding_or_model_load(self) -> None:
        with patch(
            "openai.resources.embeddings.Embeddings.create"
        ) as embeddings_create:
            result = Scorer(enable_routing_log=False).classify("hello world")

        embeddings_create.assert_not_called()
        assert isinstance(result, ClassificationResult)
        assert result.signals["method"]["raw"] == "heuristic-features"
        assert result.signals["featureVersion"] == "v1"
        assert isinstance(result.signals["semanticLabels"], dict)
        assert len(result.dimensions) == 15
        assert result.tier in {Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX, Tier.REASONING}

    def test_agentic_score_is_derived_from_agentic_task_dimension(self) -> None:
        result = Scorer(enable_routing_log=False).classify("fix this bug and run tests")

        assert result.agentic_score == 1.0
        assert result.tier in {Tier.MEDIUM, Tier.COMPLEX, Tier.REASONING}

    def test_reasoning_prompt_routes_to_reasoning(self) -> None:
        result = Scorer(enable_routing_log=False).classify(
            "prove why this algorithm is correct and explain the root cause"
        )

        assert result.tier == Tier.REASONING
        assert result.signals["semanticLabels"]["reasoningMarkers"] == "high"

    def test_simple_prompt_still_has_full_dimension_shape(self) -> None:
        result = Scorer(enable_routing_log=False).classify("hello")

        assert result.signals["method"]["raw"] == "heuristic-features"
        assert result.confidence > 0.0
        assert result.score >= 0.0
        assert len(result.dimensions) == 15

    def test_configured_fallback_when_runtime_classifier_fails(self) -> None:
        config = ScoringConfig(fallback_tier=Tier.MEDIUM, fallback_confidence=0.31)
        with patch(
            "kani.scorer._heuristic_semantic_labels", side_effect=RuntimeError("boom")
        ):
            result = Scorer(config=config, enable_routing_log=False).classify(
                "anything"
            )

        assert result.tier == Tier.MEDIUM
        assert result.confidence == 0.31
        assert result.score == 0.0
        assert result.signals["method"]["raw"] == "default"
        assert result.agentic_score == 0.0
        assert result.dimensions == []
