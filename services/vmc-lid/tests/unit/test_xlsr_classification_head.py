import pytest
import numpy as np
from core.classification_head import (
    NoClassificationHead,
    LinearClassificationHead,
    MLPClassificationHead,
    ClassificationHeadNotAvailableError,
)
from routing.canonical_registry import get_canonical_registry


class TestXLSRClassificationHead:

    def test_missing_classification_head(self):
        """Verifies NoClassificationHead raises ClassificationHeadNotAvailableError."""
        head = NoClassificationHead()
        assert head.is_available is False
        assert len(head.supported_languages) == 19

        dummy_emb = np.zeros((1, 1024), dtype=np.float32)
        with pytest.raises(ClassificationHeadNotAvailableError):
            head.predict(dummy_emb)

    def test_canonical_label_order(self):
        """Verifies that classification head derives labels from canonical registry in exact order."""
        reg = get_canonical_registry()
        labels = reg.lid_labels()
        assert len(labels) == 19
        assert labels[0] == "te-IN"
        assert labels[-1] == "en-IN"
        assert "kut-IN" in labels
        assert "kfr-IN" not in labels

        head = NoClassificationHead(registry=reg)
        assert head.supported_languages == labels

    def test_invalid_probability_vector(self):
        """Verifies dimension validation on LinearClassificationHead."""
        weights = np.zeros((1024, 10), dtype=np.float32)  # Wrong output dim: 10 instead of 19
        bias = np.zeros(10, dtype=np.float32)

        with pytest.raises(ValueError) as exc:
            LinearClassificationHead(weights=weights, bias=bias)
        assert "Weight dimension mismatch" in str(exc.value)

    def test_mlp_classification_head(self):
        """Verifies MLPClassificationHead forward prediction across 19 canonical classes."""
        reg = get_canonical_registry()
        # 2 layers: 1024 -> 256 -> 19
        w1 = np.random.randn(1024, 256).astype(np.float32) * 0.01
        b1 = np.zeros(256, dtype=np.float32)
        w2 = np.random.randn(256, 19).astype(np.float32) * 0.01
        b2 = np.zeros(19, dtype=np.float32)
        # Bias towards Hindi (hi-IN, index 4)
        b2[4] = 10.0

        mlp_head = MLPClassificationHead(layers=[(w1, b1), (w2, b2)], registry=reg)
        assert mlp_head.is_available is True

        emb = np.zeros((1, 10, 1024), dtype=np.float32)
        top_lang, conf, probs = mlp_head.predict(emb)
        assert top_lang == "hi-IN"
        assert conf > 0.9
        assert len(probs) == 19
        assert "hi-IN" in probs
        assert "kut-IN" in probs
