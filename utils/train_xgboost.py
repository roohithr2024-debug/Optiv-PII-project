"""Train the optional XGBoost PII corroboration model from Excel annotations.

Usage:
    python utils/train_xgboost.py ^
      --training-set "C:\\path\\Training_Set.xlsx" ^
      --testing-set "C:\\path\\Testing_Set.xlsx"
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

logger = logging.getLogger(__name__)

ANNOTATION_TYPES = {
    "name": "PERSON",
    "credit_card": "CREDIT_CARD",
    "email": "EMAIL",
    "phone": "PHONE",
    "address": "ADDRESS",
    "ssn": "GOVERNMENT_ID",
    "company": "OTHER_PII",
    "url": "OTHER_PII",
}


def _annotations(value: object) -> List[Tuple[int, int, str]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    result = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            start, end, label = item
            if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                result.append((start, end, label.lower()))
    return result


def _build_examples(frame: pd.DataFrame, negative_ratio: int = 1) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    rng = random.Random(42)

    for row in frame[["Text", "True Predictions"]].itertuples(index=False):
        document = row[0] if isinstance(row[0], str) else ""
        annotations = _annotations(row[1])
        if not document:
            continue

        spans = [(start, end) for start, end, _ in annotations]
        for start, end, label in annotations:
            value = document[start:end].strip()
            mapped = ANNOTATION_TYPES.get(label)
            if value and mapped:
                texts.append(value)
                labels.append(mapped)

        # Hard negatives are short chunks from non-annotated text. They keep
        # the model from treating every ordinary word as a PII entity.
        for _ in range(min(negative_ratio, max(1, len(annotations)))):
            words = document.split()
            if not words:
                continue
            for _attempt in range(10):
                start_word = rng.randrange(len(words))
                candidate = words[start_word].strip(".,:;()[]{}\"'")
                if not candidate:
                    continue
                char_start = document.find(candidate)
                char_end = char_start + len(candidate)
                if not any(char_start < end and char_end > start for start, end in spans):
                    texts.append(candidate)
                    labels.append("NONE")
                    break

    return texts, labels


def train(training_set: Path, testing_set: Path, output: Path) -> Dict[str, object]:
    train_frame = pd.read_excel(training_set, engine="openpyxl", usecols=["Text", "True Predictions"])
    test_frame = pd.read_excel(testing_set, engine="openpyxl", usecols=["Text", "True Predictions"])
    train_texts, train_labels = _build_examples(train_frame)
    test_texts, test_labels = _build_examples(test_frame)

    if len(set(train_labels)) < 2:
        raise ValueError("Training annotations must contain at least two classes.")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=180,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, y_train)

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "model": model, "labels": encoder.classes_.tolist()},
        output,
        compress=3,
    )

    metrics: Dict[str, object] = {
        "training_examples": len(train_texts),
        "classes": encoder.classes_.tolist(),
        "artifact": str(output),
    }
    if test_texts:
        predicted = encoder.inverse_transform(model.predict(vectorizer.transform(test_texts)).astype(int))
        metrics["testing_report"] = classification_report(
            test_labels,
            predicted,
            labels=encoder.classes_.tolist(),
            zero_division=0,
            output_dict=True,
        )
    metrics_path = output.with_name("xgboost_ml_metrics.json")
    metrics["metrics_file"] = str(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", type=Path, required=True)
    parser.add_argument("--testing-set", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "models"
        / "artifacts"
        / "xgboost_pii_classifier.joblib",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    metrics = train(args.training_set, args.testing_set, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
