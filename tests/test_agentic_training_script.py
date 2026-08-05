from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from kani.config import EmbeddingConfig, KaniConfig, ProviderConfig
from kani.feature_training import (
    build_embedding_client,
    load_feature_examples,
    load_or_compute_embeddings,
    train_feature_classifier,
)


def _row(agentic: str) -> dict[str, str]:
    return {
        "prompt": f"prompt-{agentic}",
        "codePresence": "low",
        "reasoningMarkers": "medium",
        "technicalTerms": "medium",
        "creativeMarkers": "low",
        "simpleIndicators": "high",
        "multiStepPatterns": "medium",
        "questionComplexity": "medium",
        "imperativeVerbs": "low",
        "constraintCount": "medium",
        "outputFormat": "low",
        "referenceComplexity": "medium",
        "negationComplexity": "low",
        "domainSpecificity": "medium",
        "agenticTask": agentic,
    }


def test_load_feature_examples_reads_semantic_dimensions(tmp_path: Path) -> None:
    data_path = tmp_path / "distilled_feature_dataset.json"
    data_path.write_text(
        json.dumps([_row("low"), _row("high")]),
        encoding="utf-8",
    )

    request_texts, context_texts, labels_by_dimension = load_feature_examples(data_path)

    # Backward-compat: request falls back to prompt; context falls back to empty.
    assert request_texts == ["prompt-low", "prompt-high"]
    assert context_texts == ["", ""]
    assert labels_by_dimension["agenticTask"] == ["low", "high"]


def test_load_feature_examples_dual_inputs(tmp_path: Path) -> None:
    data_path = tmp_path / "distilled_feature_dataset.json"
    rows = []
    for agentic, request, context in (
        ("low", "What is recursion?", "user: Hi\nassistant: Hello!"),
        ("high", "Write a trie in Python", "user: I need a data structure"),
    ):
        row = _row(agentic)
        row["last_user_message"] = request
        row["context_text"] = context
        rows.append(row)
    data_path.write_text(json.dumps(rows), encoding="utf-8")

    request_texts, context_texts, labels_by_dimension = load_feature_examples(data_path)

    assert request_texts == ["What is recursion?", "Write a trie in Python"]
    assert context_texts == [
        "user: Hi\nassistant: Hello!",
        "user: I need a data structure",
    ]
    assert labels_by_dimension["agenticTask"] == ["low", "high"]


def test_load_feature_examples_backward_compat(tmp_path: Path) -> None:
    data_path = tmp_path / "distilled_feature_dataset.json"
    data_path.write_text(
        json.dumps([_row("low"), _row("high")]),
        encoding="utf-8",
    )

    request_texts, context_texts, _labels = load_feature_examples(data_path)

    assert request_texts == ["prompt-low", "prompt-high"]
    assert context_texts == ["", ""]


def test_load_feature_examples_rejects_invalid_labels(tmp_path: Path) -> None:
    data_path = tmp_path / "bad_distilled_feature_dataset.json"
    bad = _row("low")
    bad["reasoningMarkers"] = "invalid"
    data_path.write_text(json.dumps([bad, _row("high")]), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid label for reasoningMarkers"):
        load_feature_examples(data_path)


def test_load_or_compute_embeddings_cache_key_includes_model(tmp_path: Path) -> None:
    class _EmbeddingsAPI:
        def __init__(self, vector: list[float]) -> None:
            self._vector = vector

        def create(self, input: list[str], model: str) -> object:
            return type(
                "Resp",
                (),
                {
                    "data": [
                        type("Item", (), {"embedding": self._vector}) for _ in input
                    ]
                },
            )()

    class _Client:
        def __init__(self, vector: list[float]) -> None:
            self.embeddings = _EmbeddingsAPI(vector)

    cache_dir = tmp_path / "cache"
    texts = ["hello"]

    first = load_or_compute_embeddings(_Client([1.0, 2.0]), texts, cache_dir, "model-a")
    second = load_or_compute_embeddings(
        _Client([3.0, 4.0]), texts, cache_dir, "model-b"
    )

    assert first.tolist() == [[1.0, 2.0]]
    assert second.tolist() == [[3.0, 4.0]]
    assert len(list(cache_dir.glob("embeddings_*.npy"))) == 2


def test_build_embedding_client_raises_when_embedding_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = KaniConfig(
        providers={
            "default": ProviderConfig(
                name="default",
                base_url="https://api.example.com/v1",
                api_key="test-key",
            )
        },
        default_provider="default",
        embedding=EmbeddingConfig(enabled=False),
    )
    monkeypatch.setattr("kani.feature_training.load_config", lambda: cfg)

    with pytest.raises(RuntimeError, match="Embedding is disabled in config"):
        build_embedding_client()


def test_build_embedding_client_uses_enabled_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = KaniConfig(
        embedding=EmbeddingConfig(
            enabled=True,
            model="embedding-model",
            base_url="https://embeddings.example/v1",
            api_key="embedding-key",
        )
    )
    monkeypatch.setattr("kani.feature_training.load_config", lambda: cfg)

    client, model = build_embedding_client()

    assert model == "embedding-model"
    assert str(client.base_url) == "https://embeddings.example/v1/"


def test_train_feature_classifier_writes_model_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "distilled_feature_dataset.json"
    rows = []
    # ensure each dimension has at least two classes and enough samples for
    # StratifiedShuffleSplit early-stopping validation (test_size >= n_classes)
    for index in range(20):
        row = {
            "prompt": f"prompt-{index}",
            "codePresence": "high" if index % 2 else "low",
            "reasoningMarkers": "high" if index % 3 == 0 else "medium",
            "technicalTerms": "high" if index % 2 else "low",
            "creativeMarkers": "medium" if index % 2 else "low",
            "simpleIndicators": "high" if index % 2 == 0 else "low",
            "multiStepPatterns": "high" if index % 2 else "low",
            "questionComplexity": "high" if index % 3 == 0 else "low",
            "imperativeVerbs": "high" if index % 2 else "low",
            "constraintCount": "high" if index % 2 else "low",
            "outputFormat": "high" if index % 2 else "low",
            "referenceComplexity": "high" if index % 2 else "low",
            "negationComplexity": "high" if index % 2 else "low",
            "domainSpecificity": "high" if index % 2 else "low",
            "agenticTask": "high" if index % 2 else "low",
        }
        rows.append(row)

    data_path.write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setattr(
        "kani.feature_training.load_or_compute_embeddings",
        lambda client, texts, cache_path, model: np.array(
            [[float(i), float(i + 1)] for i, _ in enumerate(texts)],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "kani.feature_training.build_embedding_client",
        lambda: (object(), "test-embedding-model"),
    )

    model_path = train_feature_classifier(
        data_path=data_path,
        output_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
    )

    assert model_path == tmp_path / "models" / "feature_classifier.pkl"
    assert model_path.exists()

    bundle = pickle.loads(model_path.read_bytes())
    assert bundle["embedding_model"] == "test-embedding-model"
    assert bundle["embedding_mode"] == "dual"
    # Per-side embedding dimension (mock returns dim-2 vectors; X is 2*2=4 wide)
    assert bundle["embedding_dim"] == 2
    assert bundle["training_size"] == len(rows)
    assert bundle["feature_schema_version"] == "v1"
    assert "agenticTask" in bundle["label_encoders"]
    assert "weights" in bundle


def test_train_feature_classifier_dual_embedding_interleaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "distilled_feature_dataset.json"
    rows = []
    for index in range(20):
        row = {
            "prompt": f"prompt-{index}",
            "last_user_message": f"request-{index}",
            "context_text": f"context-{index}",
            "codePresence": "high" if index % 2 else "low",
            "reasoningMarkers": "high" if index % 3 == 0 else "medium",
            "technicalTerms": "high" if index % 2 else "low",
            "creativeMarkers": "medium" if index % 2 else "low",
            "simpleIndicators": "high" if index % 2 == 0 else "low",
            "multiStepPatterns": "high" if index % 2 else "low",
            "questionComplexity": "high" if index % 3 == 0 else "low",
            "imperativeVerbs": "high" if index % 2 else "low",
            "constraintCount": "high" if index % 2 else "low",
            "outputFormat": "high" if index % 2 else "low",
            "referenceComplexity": "high" if index % 2 else "low",
            "negationComplexity": "high" if index % 2 else "low",
            "domainSpecificity": "high" if index % 2 else "low",
            "agenticTask": "high" if index % 2 else "low",
        }
        rows.append(row)
    data_path.write_text(json.dumps(rows), encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def _capture_embeddings(
        client: object, texts: list[str], cache_path: Path, model: str
    ) -> np.ndarray:
        captured["texts"] = texts
        return np.array(
            [[float(i), float(i + 1)] for i, _ in enumerate(texts)],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "kani.feature_training.load_or_compute_embeddings", _capture_embeddings
    )
    monkeypatch.setattr(
        "kani.feature_training.build_embedding_client",
        lambda: (object(), "test-embedding-model"),
    )

    train_feature_classifier(
        data_path=data_path,
        output_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
    )

    # Verify interleaved order: [request_1, context_1, request_2, context_2, ...]
    texts = captured["texts"]
    assert len(texts) == 40
    for i in range(20):
        assert texts[2 * i] == f"request-{i}"
        assert texts[2 * i + 1] == f"context-{i}"
