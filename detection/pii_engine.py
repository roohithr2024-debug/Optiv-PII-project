"""Hybrid PII Detection Engine combining Regex, NER, and Context Rules.

Merges overlapping findings, calculates confidence scores, boosts confidence on multi-method
agreement, and applies internal risk severity classification.
"""
import logging
from typing import List, Dict, Tuple

from models.schemas import (
    DocumentBlock,
    PIIFinding,
    PIIType,
    Severity,
    ConfidenceLevel,
)
from detection.regex_detector import RegexDetector
from detection.ner_detector import NERDetector
from detection.context_rules import ContextDetector
from detection.xgboost_detector import XGBoostPIICorroborator
from utils.helpers import mask_sensitive_value

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
}


class PIIEngine:
    """Orchestrates hybrid PII detection across document blocks."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.regex_detector = RegexDetector()
        self.ner_detector = NERDetector(model_name=spacy_model)
        self.context_detector = ContextDetector()
        self.ml_corroborator = XGBoostPIICorroborator()

    def detect_in_block(self, block: DocumentBlock) -> List[PIIFinding]:
        """Run all detectors on a single DocumentBlock and merge overlapping results."""
        raw_findings: List[PIIFinding] = []

        # 1. Regex Detection
        raw_findings.extend(
            self.regex_detector.detect(
                text=block.text,
                document_id=block.document_id,
                file_name=block.file_name,
                location=block.location,
                block_order=block.order,
            )
        )

        # 2. NER Detection
        raw_findings.extend(
            self.ner_detector.detect(
                text=block.text,
                document_id=block.document_id,
                file_name=block.file_name,
                location=block.location,
                block_order=block.order,
            )
        )

        # 3. Context Rules Detection
        raw_findings.extend(
            self.context_detector.detect(
                text=block.text,
                document_id=block.document_id,
                file_name=block.file_name,
                location=block.location,
                block_order=block.order,
            )
        )

        # Merge overlapping findings
        merged = self._merge_findings(raw_findings, block.text)
        for finding in merged:
            self.ml_corroborator.corroborate(finding)
        return merged

    def detect_blocks(self, blocks: List[DocumentBlock]) -> List[PIIFinding]:
        """Detect PII across all document blocks, returning a unified list."""
        all_findings: List[PIIFinding] = []
        for block in blocks:
            findings = self.detect_in_block(block)
            all_findings.extend(findings)
        
        # Sort by block order and character offset
        all_findings.sort(key=lambda f: (f.block_order, f.start))
        return all_findings

    def detect_raw_text(
        self,
        text: str,
        document_id: str = "doc_validation",
        file_name: str = "validation.txt",
        location: str = "sanitized_stream"
    ) -> List[PIIFinding]:
        """Direct detection on raw text (used for validation and testing)."""
        temp_block = DocumentBlock(
            document_id=document_id,
            file_name=file_name,
            file_type="txt",
            location=location,
            block_type="text",
            text=text,
            order=0,
        )
        return self.detect_in_block(temp_block)

    def _merge_findings(self, findings: List[PIIFinding], full_text: str) -> List[PIIFinding]:
        """Merge overlapping or identical findings in the same block."""
        if not findings:
            return []

        # Sort by start offset ascending, then length descending
        sorted_findings = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
        merged: List[PIIFinding] = []

        for current in sorted_findings:
            if not merged:
                merged.append(current)
                continue

            last = merged[-1]

            # Check for overlap: max(start1, start2) < min(end1, end2)
            has_overlap = max(last.start, current.start) < min(last.end, current.end)

            if has_overlap:
                # Merge current into last
                # 1. Combine methods
                combined_methods = list(dict.fromkeys(last.detection_method + current.detection_method))
                
                # 2. Boost confidence when multiple methods agree
                base_conf = max(last.confidence, current.confidence)
                if len(combined_methods) > 1:
                    boosted_conf = min(0.99, base_conf + 0.05)
                else:
                    boosted_conf = base_conf

                # 3. Determine highest severity
                last_sev_rank = SEVERITY_ORDER.get(last.severity, 1)
                curr_sev_rank = SEVERITY_ORDER.get(current.severity, 1)
                highest_severity = last.severity if last_sev_rank >= curr_sev_rank else current.severity

                # 4. Resolve PII Type: Context or Regex often more specific than generic NER
                # If one is PERSON and other is EMPLOYEE_ID with CONTEXT, EMPLOYEE_ID or the higher confidence wins
                if last.confidence >= current.confidence:
                    resolved_type = last.pii_type
                    best_explanation = last.explanation
                else:
                    resolved_type = current.pii_type
                    best_explanation = current.explanation

                # If methods combined, update explanation
                if len(combined_methods) > 1:
                    method_str = " + ".join(combined_methods)
                    best_explanation = f"Cross-verified by {method_str}: {best_explanation}"

                # 5. Span boundaries (pick the span that best bounds the entity text)
                new_start = min(last.start, current.start)
                new_end = max(last.end, current.end)
                merged_text = full_text[new_start:new_end].strip()

                # Update confidence level
                conf_level = (
                    ConfidenceLevel.HIGH if boosted_conf >= 0.90
                    else (ConfidenceLevel.MEDIUM if boosted_conf >= 0.70 else ConfidenceLevel.LOW)
                )

                # Mutate last in place
                last.start = new_start
                last.end = new_end
                last.text = merged_text if merged_text else last.text
                last.pii_type = resolved_type
                last.confidence = boosted_conf
                last.confidence_level = conf_level
                last.detection_method = combined_methods
                last.severity = highest_severity
                last.explanation = best_explanation
                last.masked_text = mask_sensitive_value(last.text, resolved_type.value)
            else:
                merged.append(current)

        return merged
