"""Optional XGBoost corroboration for findings produced by the hybrid engine.

The rule, regex, and NER detectors remain the source of candidate spans. This
module only adds a trained-data signal when a local model artifact exists and
the model agrees with the candidate's existing PII type.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models.schemas import PIIFinding, PIIType

logger = logging.getLogger(__name__)


class XGBoostPIICorroborator:
    """Load and apply an optional local XGBoost text classifier."""

    def __init__(
        self,
        artifact_path: Optional[Path] = None,
        minimum_probability: float = 0.70,
    ) -> None:
        self.artifact_path = artifact_path or (
            Path(__file__).resolve().parent.parent
            / "models"
            / "artifacts"
            / "xgboost_pii_classifier.joblib"
        )
        self.minimum_probability = minimum_probability
        self._bundle: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self._load()

    @property
    def available(self) -> bool:
        return self._bundle is not None

    def _load(self) -> None:
        if not self.artifact_path.exists():
            return
        try:
            import joblib

            bundle = joblib.load(self.artifact_path)
            if not isinstance(bundle, dict) or not {"vectorizer", "model", "labels"} <= set(bundle):
                raise ValueError("XGBoost artifact is missing required bundle fields.")
            self._bundle = bundle
        except Exception as exc:  # pragma: no cover - depends on optional artifact
            self._load_error = str(exc)
            logger.warning("XGBoost artifact unavailable: %s", exc)

    def score_finding(self, finding: PIIFinding) -> Optional[Tuple[str, float]]:
        """Return the predicted class and probability for a candidate finding."""
        if not self._bundle:
            return None

        try:
            vectorizer = self._bundle["vectorizer"]
            model = self._bundle["model"]
            labels = self._bundle["labels"]
            probabilities = model.predict_proba(vectorizer.transform([finding.text]))[0]
            best_index = int(probabilities.argmax())
            return str(labels[best_index]), float(probabilities[best_index])
        except Exception as exc:  # pragma: no cover - defensive artifact boundary
            logger.warning("XGBoost scoring skipped: %s", exc)
            return None

    def corroborate(self, finding: PIIFinding) -> None:
        """Add an agreeing model signal without changing existing classification."""
        result = self.score_finding(finding)
        if result is None:
            return

        predicted_type, probability = result
        expected_type = finding.pii_type.value if isinstance(finding.pii_type, PIIType) else str(finding.pii_type)
        if predicted_type != expected_type or probability < self.minimum_probability:
            return

        if "XGBOOST" not in finding.detection_method:
            finding.detection_method.append("XGBOOST")
        finding.confidence = min(0.99, max(finding.confidence, probability))

