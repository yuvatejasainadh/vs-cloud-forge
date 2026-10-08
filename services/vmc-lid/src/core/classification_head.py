from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

from routing.canonical_registry import CanonicalRegistry, get_canonical_registry
from telemetry.logging import get_logger

logger = get_logger("classification_head")


class ClassificationHeadNotAvailableError(Exception):
    """Raised when inference is requested on a backbone or runtime with no trained classification head."""
    pass


class LIDClassificationHead(ABC):
    """
    Abstract pluggable classification head interface for mapping backbone speech embeddings
    to canonical 19-class language identification distributions.
    """

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Takes raw backbone embeddings (e.g., shape (1, hidden_dim) or (1, time, hidden_dim)),
        and returns:
          (top_language_code, top_confidence, all_19_probabilities)
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if a valid trained checkpoint is loaded and ready."""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Returns the list of target language codes produced by this head."""
        pass


class NoClassificationHead(LIDClassificationHead):
    """
    Represents an unhydrated or absent classification head (e.g. for base XLS-R 300M backbone).
    Guarantees no synthetic, untrained, or random predictions are ever produced.
    """

    def __init__(self, registry: Optional[CanonicalRegistry] = None):
        self._registry = registry or get_canonical_registry()

    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        raise ClassificationHeadNotAvailableError(
            "No trained 19-language LID classification head is available for this backbone. "
            "Inference cannot produce language predictions without a validated trained checkpoint."
        )

    @property
    def is_available(self) -> bool:
        return False

    @property
    def supported_languages(self) -> List[str]:
        return self._registry.lid_labels()


class LinearClassificationHead(LIDClassificationHead):
    """
    Linear projection classification head (e.g. W * pooled_emb + b).
    Requires explicit pre-trained weight matrices.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        registry: Optional[CanonicalRegistry] = None
    ):
        self._registry = registry or get_canonical_registry()
        self._labels = self._registry.lid_labels()
        
        if weights.shape[1] != len(self._labels):
            raise ValueError(
                f"Weight dimension mismatch: expected output dim {len(self._labels)}, got {weights.shape[1]}"
            )
        if len(bias) != len(self._labels):
            raise ValueError(
                f"Bias dimension mismatch: expected output dim {len(self._labels)}, got {len(bias)}"
            )

        self._weights = weights
        self._bias = bias
        self._is_available = True

    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if not self._is_available:
            raise ClassificationHeadNotAvailableError("Linear head weights not loaded.")

        # Pool temporal dimension if necessary: (batch, time, dim) -> (batch, dim)
        if embeddings.ndim == 3:
            pooled = np.mean(embeddings, axis=1)
        elif embeddings.ndim == 2:
            pooled = embeddings
        else:
            raise ValueError(f"Expected 2D or 3D embedding array, got shape {embeddings.shape}")

        logits = np.dot(pooled, self._weights) + self._bias  # shape: (batch, num_classes)
        logits_1d = logits[0] if logits.ndim > 1 else logits

        # Numerically stable softmax
        shifted = logits_1d - np.max(logits_1d)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l)

        probabilities = {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}
        top_idx = int(np.argmax(probs))
        top_lang = self._labels[top_idx]
        confidence = float(probs[top_idx])

        return top_lang, confidence, probabilities

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def supported_languages(self) -> List[str]:
        return self._labels


class MLPClassificationHead(LIDClassificationHead):
    """Multi-Layer Perceptron classification head."""

    def __init__(self, layers: List[Tuple[np.ndarray, np.ndarray]], registry: Optional[CanonicalRegistry] = None):
        self._registry = registry or get_canonical_registry()
        self._labels = self._registry.lid_labels()
        self._layers = layers
        self._is_available = len(layers) > 0

    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if not self._is_available:
            raise ClassificationHeadNotAvailableError("MLP head weights not loaded.")

        h = np.mean(embeddings, axis=1) if embeddings.ndim == 3 else embeddings
        for i, (w, b) in enumerate(self._layers):
            h = np.dot(h, w) + b
            if i < len(self._layers) - 1:
                # ReLU activation for hidden layers
                h = np.maximum(0, h)

        logits_1d = h[0] if h.ndim > 1 else h
        shifted = logits_1d - np.max(logits_1d)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l)

        probabilities = {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}
        top_idx = int(np.argmax(probs))
        top_lang = self._labels[top_idx]
        confidence = float(probs[top_idx])

        return top_lang, confidence, probabilities

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def supported_languages(self) -> List[str]:
        return self._labels


class ONNXClassificationHead(LIDClassificationHead):
    """ONNX-based classification head."""

    def __init__(self, onnx_model_path: Optional[str] = None, registry: Optional[CanonicalRegistry] = None):
        self._registry = registry or get_canonical_registry()
        self._labels = self._registry.lid_labels()
        self._session = None
        self._is_available = False

        if onnx_model_path:
            import os
            import onnxruntime as ort
            if os.path.exists(onnx_model_path):
                self._session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
                self._is_available = True

    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if not self._is_available or self._session is None:
            raise ClassificationHeadNotAvailableError("ONNX head model session is not loaded.")

        inputs = {self._session.get_inputs()[0].name: embeddings.astype(np.float32)}
        outputs = self._session.run(None, inputs)
        logits_1d = outputs[0][0]

        shifted = logits_1d - np.max(logits_1d)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l)

        probabilities = {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}
        top_idx = int(np.argmax(probs))
        top_lang = self._labels[top_idx]
        confidence = float(probs[top_idx])

        return top_lang, confidence, probabilities

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def supported_languages(self) -> List[str]:
        return self._labels


class TorchScriptClassificationHead(LIDClassificationHead):
    """Placeholder for future TorchScript compiled classification head."""

    def __init__(self, script_path: Optional[str] = None, registry: Optional[CanonicalRegistry] = None):
        self._registry = registry or get_canonical_registry()
        self._labels = self._registry.lid_labels()
        self._is_available = False

    def predict(self, embeddings: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        raise ClassificationHeadNotAvailableError("TorchScript classification head is not configured.")

    @property
    def is_available(self) -> bool:
        return False

    @property
    def supported_languages(self) -> List[str]:
        return self._labels
