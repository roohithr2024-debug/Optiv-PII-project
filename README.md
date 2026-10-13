# PII Security Gateway

> **Pre-LLM Security Gateway & Document Sanitization Pipeline**  
> *Core Security Principle: "NO PII → SAFE FOR DOWNSTREAM AI"*

---

## 1. Project Overview

The **PII Security Gateway** is an enterprise-grade pre-LLM cybersecurity application designed to inspect, classify, redact, and validate business evidence documents before they are submitted to downstream Large Language Models (LLMs) or generative AI services. 

Operating entirely **locally** with **zero external API calls**, the gateway guarantees that no sensitive personal data escapes the trust perimeter while maintaining the structural context of business artifacts.

---

## 2. The Problem

Enterprise business evidence artifacts (contracts, customer tickets, financial presentations, executive briefings, incident reports) frequently contain Personally Identifiable Information (PII) such as personal names, email addresses, phone numbers, corporate employee IDs, social security numbers, government IDs, and payment card numbers.

Sending unredacted evidence directly to third-party or hosted LLMs creates:
- **Regulatory Non-Compliance:** Violations of GDPR, HIPAA, CCPA, and DPDP.
- **Data Leakage Risks:** Exposure of proprietary employee identifiers, customer records, and network topologies.
- **Model Training Ingestion:** Risk of sensitive entity memorization by foundational models.

---

## 3. The Solution

The PII Security Gateway enforces a **fail-closed** security gate prior to AI consumption:
1. **Structural Document Ingestion:** Normalizes multi-format documents (PDF, DOCX, PPTX, TXT) into structured text blocks while preserving section, slide, page, and table provenance.
2. **Hybrid PII Detection:** Combines deterministic Regex, spaCy Named Entity Recognition (NER), and contextual rule-matching with cross-corroboration confidence boosting.
3. **Consistent Semantic Redaction:** Replaces detected PII with semantic placeholders (e.g., `[PERSON_001]`, `[EMAIL_001]`) rather than blunt black-box deletions, preserving relational reasoning for downstream AI.
4. **Post-Redaction Validation:** Automatically re-scans the sanitized text with the detection engine. If *any* residual PII is detected, the system immediately **fails closed** and marks the document as **BLOCKED**.

---

## 4. Architecture & Pipeline

```text
User
  │
  ▼
File Upload (PDF, DOCX, PPTX, TXT)
  │
  ▼
File Classification & Dispatch
  │
  ▼
Document Extraction (PyMuPDF, python-docx, python-pptx, txt)
  │
  ▼
Normalized Document Blocks (Structure & Source Preserved)
  │
  ▼
Hybrid PII Detection Engine
  ├── A. Regex Detection (Email, Phone, CC, IP, SSN, Govt ID)
  ├── B. NER Detection (spaCy PERSON, GPE, Date)
  └── C. Context Rules (Employee ID:, Contact:, DOB:, etc.)
  │
  ▼
PII Classification & Exposure Severity Assessment
  │
  ▼
Exposure Report & Traceability Matrix
  │
  ▼
Redaction Engine (Consistent Document-Scoped Semantic Placeholders)
  │
  ▼
Post-Redaction Validation (Re-scan of Sanitized Blocks)
  │
  ▼
Security Decision:
  ├── PASS (Zero Residual PII) ──► STATUS: SANITIZED (Safe for AI)
  └── FAIL (Residual PII Found) ──► STATUS: BLOCKED (Fail-Closed)
  │
  ▼
Sanitized Content Output & Compliance Audit Logs (JSON/CSV)
```

---

## 5. Supported Document Formats

| Format | Parsing Library | Structural Information Preserved | Scanned Handling |
| :--- | :--- | :--- | :--- |
| **PDF** | PyMuPDF (`fitz`) | Page boundaries, text blocks, reading order | Flags pages `< 15` chars as `SCANNED_PAGE_REQUIRES_OCR`; modular Tesseract fallback |
| **DOCX** | `python-docx` | Headings, paragraph hierarchies, tables, rows | Native XML structure extraction |
| **PPTX** | `python-pptx` | Slides, text shapes, tables, speaker notes | Multi-slide shape traversal |
| **TXT** | Native Python | Paragraph boundaries, line indexing | Multi-encoding fallback (UTF-8, Latin-1) |

---

## 6. PII Detection Methodology

The gateway employs a three-tier **Hybrid Detection Engine**:

1. **Regex Detectors (`regex_detector.py`):**
   - **EMAIL:** RFC-compliant mailbox pattern.
   - **PHONE:** International formats (`+1`, `+91`), US standard `(212) 555-1234`, and domestic 10-digit mobile patterns.
   - **EMPLOYEE_ID:** Corporate patterns (`EMP10293`, `E10234`, `EMP-99214`, `STAFF-1234`).
   - **CREDIT_CARD:** 13-19 digit card patterns with algorithmic **Luhn checksum validation**.
   - **IP_ADDRESS:** IPv4 addresses with 0-255 boundary verification.
   - **GOVERNMENT_ID:** US SSN (`XXX-XX-XXXX`), Indian Aadhaar (`XXXX XXXX XXXX`), and generic government identifiers (`GVT-XXXXXX`).

2. **Named Entity Recognition (`ner_detector.py`):**
   - Employs spaCy's `en_core_web_sm` model.
   - Detects `PERSON` entities with high confidence.
   - Contextually screens `GPE`/`LOC` (residential context) and `DATE` (birth date cues) to prevent false positives on general dates and locations.
   - Graceful fallback: If spaCy is unavailable, the application continues safely using regex and context rules without crashing.

3. **Context-Sensitive Rules (`context_rules.py`):**
   - Matches evidence prefix cues such as `"Employee ID:"`, `"Assigned To:"`, `"Contact:"`, `"Date of Birth:"`, `"Government ID:"`.
   - Boosts detection confidence up to **0.99** when explicit evidence labels corroborate findings.

4. **Duplicate & Overlap Merging:**
   - Spans matched by multiple detectors are unified into a single finding.
   - Detection methods are aggregated (e.g., `["NER", "CONTEXT"]`).
   - Confidence is boosted upon multi-detector agreement.

---

## 7. Redaction & Consistent Placeholders

Unlike crude redactions that replace all text with `[REDACTED]`, this gateway generates **transparent, semantic, document-consistent placeholders**:
- First person encountered: `[PERSON_001]`
- Second person encountered: `[PERSON_002]`
- Recurring occurrences of the first person: Always mapped to `[PERSON_001]`
- Email: `[EMAIL_001]`
- Phone: `[PHONE_001]`
- Employee ID: `[EMPLOYEE_ID_001]`

This allows downstream AI models to understand relationships (e.g., *"Did `[PERSON_001]` contact `[PERSON_002]`?"*) without ever exposing personal identifiable values. The mapping table is stored strictly in the temporary in-memory session and is **never** transmitted to an LLM.

---

## 8. Post-Redaction Validation & Fail-Closed Security

A fundamental security principle of this gateway is:  
**Never trust redaction execution without independent verification.**

After semantic redaction:
1. The sanitized text blocks are fed back into the Hybrid PII Detection Engine for an independent **Re-Scan**.
2. If any high-confidence PII remains (residual PII count > 0), the document is marked **BLOCKED**.
3. If an extraction error, corrupt file, or empty document occurs, the system **fails closed** and blocks the document.
4. Only documents with **zero detectable PII** upon re-scan receive the **SANITIZED** status.

---

## 9. Risk Classification Severity Model

The application uses an internal risk severity classification:
- **CRITICAL:** Government IDs (SSN, Aadhaar, National ID), Credit Card numbers.
- **HIGH:** Corporate Employee IDs, Personal Emails, Phone Numbers, Person names.
- **MEDIUM:** Residential Addresses, Dates of Birth.
- **LOW:** Internal IP Addresses, contextual signals.

---

## 10. Novel Contributions and Presentation Highlights

The project combines established detection techniques into a security-focused,
explainable, and fail-closed pre-LLM pipeline. The following capabilities are
the main novel contributions to highlight during a presentation.

### 10.1 Pre-LLM Privacy Enforcement

The gateway inspects and sanitizes evidence **before** it is submitted to a
downstream AI or LLM service. This changes the security question from
*"Can the model be trusted with raw data?"* to *"Has the data been proven safe
before model ingestion?"*

The system operates locally and does not require external AI or cloud APIs,
keeping raw documents inside the organization's trust perimeter.

### 10.2 Hybrid and Context-Aware PII Detection

PII detection combines:

- Deterministic regex detection for structured values such as emails, phones,
  payment cards, IP addresses, government IDs, and employee IDs.
- spaCy NER for person entities.
- Business-context rules for labels such as `Employee ID`, `Contact`,
  `Date of Birth`, `Passport`, `Government ID`, `Phone`, and `Address`.

This allows the system to use both the value itself and its surrounding
business context. A value that is ambiguous in isolation can become
high-confidence PII when it follows an explicit evidence label.

### 10.3 Cross-Detector Corroboration

When multiple detectors identify the same entity, the hybrid engine merges the
overlapping findings instead of reporting duplicates. It retains the detection
methods and boosts confidence when independent signals agree.

Every finding is therefore explainable through its:

- PII type
- Severity
- Confidence score
- Detection methods
- Document location
- Masked value
- Human-readable explanation

### 10.4 Semantic and Identity-Consistent Redaction

Instead of replacing every value with a generic `[REDACTED]` token, the
redaction engine generates semantic placeholders such as:

```text
[PERSON_001]
[EMAIL_001]
[PHONE_001]
[EMPLOYEE_ID_001]
```

Repeated occurrences of the same entity receive the same placeholder across
the document. This removes the sensitive value while preserving relationships
for downstream reasoning, for example:

```text
[PERSON_001] contacted [PERSON_002] using [EMAIL_001].
```

The internal value-to-placeholder mapping is kept in the local processing
session and is not transmitted to an external service.

### 10.5 Independent Post-Redaction Verification

The sanitizer is never trusted without a second scan. The security workflow is:

```text
Extract -> Detect -> Assess Risk -> Redact -> Re-scan -> Validate -> Release or Block
```

If actionable PII remains after redaction, the document is marked `BLOCKED`.
This protects against missed detections, incorrect replacement offsets, partial
redactions, and implementation errors.

### 10.6 Fail-Closed Security Decisions

The gateway treats uncertainty as unsafe. Empty files, corrupt files,
extraction failures, detection failures, redaction failures, and residual PII
cannot silently pass through the pipeline.

Each document receives an actionable verdict:

- `SANITIZED`: zero detectable PII remains after validation.
- `BLOCKED`: actionable PII remains or a security-stage failure occurred.
- `REVIEW_REQUIRED`: manual review is needed, such as for unreadable scanned
  content or low-confidence residual signals.

### 10.7 Structure-Preserving Multi-Format Processing

The extractors support PDF, DOCX, PPTX, and TXT evidence while preserving
provenance such as pages, slides, paragraphs, tables, sections, and block
locations. This provides traceability that would be lost by flattening every
document into an unstructured text string.

The system also detects scanned PDF pages with insufficient extractable text
and flags them for OCR or review. It does not interpret "no extracted text" as
"no PII".

### 10.8 Explainable Document Risk Scoring

The risk classifier evaluates more than the raw number of findings. It
considers:

- Sensitivity of detected categories
- Presence of government IDs or payment cards
- Number of distinct entities
- Number of distinct PII categories
- Correlation between multiple personal attributes

It produces an explainable score from 0 to 100, a risk tier
(`LOW`, `MODERATE`, `HIGH`, or `CRITICAL`), risk factors, and a recommended
action.

### 10.9 False-Positive Reduction

The detection layer validates candidates instead of accepting every pattern
match. Examples include:

- Luhn checksum validation for credit-card candidates.
- Digit and format checks for employee and government IDs.
- Realistic length and repetition checks for phone numbers.
- Filtering technical terms and document labels from person-name candidates.
- Ignoring generated placeholders during subsequent scans.

These safeguards reduce the risk of redacting harmless technical, numerical,
or structural content.

### 10.10 Audit-Ready Outputs

The dashboard produces sanitized text, JSON forensic reports, and CSV
traceability matrices. These outputs support security review, compliance
evidence, incident investigation, and integration with downstream audit or
SIEM workflows.

### 10.11 Metrics and Claims Boundary

The project implements support for precision, recall, F1 score, true positives,
false positives, false negatives, residual PII count, and structure-retention
score. However, validated precision, recall, accuracy, and F1 results require a
labeled ground-truth dataset.

Until such annotations are supplied, the system correctly reports:

```text
Metrics unavailable — ground truth not provided.
```

The current demonstrated results are operational security metrics, including
PII findings, confidence scores, risk tiers, redaction replacements,
post-redaction residual findings, validation status, and structure retention.
Confidence scores are detector heuristics and must not be presented as
statistically validated accuracy.

### 10.12 Optional XGBoost Corroboration Module

The project includes an offline XGBoost text-classification module trained from
the supplied Excel annotations. It acts as a corroboration layer for findings
already produced by the existing regex, context, and NER detectors; it does not
replace those detectors or change the frontend workflow.

To retrain the local artifact:

```bash
python utils/train_xgboost.py ^
  --training-set "C:\path\Training_Set.xlsx" ^
  --testing-set "C:\path\Testing_Set.xlsx"
```

The generated artifact is stored at
`models/artifacts/xgboost_pii_classifier.joblib`. When that artifact is absent
or cannot be loaded, the gateway continues with its original detection
behavior. When it is available and agrees with an existing finding, the
finding records `XGBOOST` as an additional detection method.

The supplied testing workbook produced the following span-classification report:

| Metric | Result |
| :--- | :--- |
| Accuracy | 100.0% |
| Weighted precision | 100.0% |
| Weighted recall | 100.0% |
| Weighted F1 | 100.0% |
| Extracted test examples | 287 |

The per-class report is saved separately at
`models/artifacts/xgboost_ml_metrics.json` and is displayed in the dashboard's
**ML Metrics** tab. These results are specific to the supplied annotated
synthetic/test distribution. They are not a claim of 100% real-world accuracy:
the evaluation classifies extracted entity spans rather than complete
documents, and generated data can contain highly regular patterns. A later
overlap audit found 30 exact annotated span values shared between the training
and testing annotations, so this score must be presented as a dataset-specific
benchmark.

### Recommended Demonstration Flow

For a project presentation, demonstrate the following sequence:

1. Upload a document containing a name, email, phone number, and employee ID.
2. Show extracted page, slide, paragraph, or table provenance.
3. Display the finding type, severity, confidence, method, and explanation.
4. Show the document-level risk score and risk tier.
5. Show consistent semantic placeholders in the sanitized output.
6. Display the independent post-redaction validation result.
7. Demonstrate a residual-PII or extraction-failure case that is blocked.
8. Export the JSON audit report or CSV traceability matrix.

---

## 11. Project Structure

```
pii-security-gateway/
├── app.py                     # Streamlit cybersecurity dashboard
├── pipeline.py                # End-to-end security pipeline coordinator
├── requirements.txt           # Dependency declarations
├── README.md                  # Comprehensive technical documentation
├── .gitignore                 # Version control exclusions
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Global paths, regex patterns, thresholds
│
├── models/
│   ├── __init__.py
│   ├── schemas.py             # Common data model (DocumentBlock, PIIFinding, etc.)
│   └── artifacts/
│       ├── xgboost_pii_classifier.joblib  # Trained local XGBoost artifact
│       └── xgboost_ml_metrics.json       # Standalone evaluation report
│
├── extractors/
│   ├── __init__.py            # File format dispatcher
│   ├── pdf_extractor.py       # PyMuPDF extractor with OCR detection
│   ├── docx_extractor.py      # python-docx extractor
│   ├── pptx_extractor.py      # python-pptx extractor
│   └── txt_extractor.py       # Plain text extractor
│
├── detection/
│   ├── __init__.py
│   ├── regex_detector.py      # Deterministic regex patterns
│   ├── ner_detector.py        # spaCy NER engine
│   ├── context_rules.py       # Context-driven keyword rules
│   ├── pii_engine.py          # Hybrid orchestrator and overlap resolver
│   └── xgboost_detector.py    # Optional ML corroboration layer
│
├── redaction/
│   ├── __init__.py
│   ├── redactor.py            # Consistent semantic placeholder generator
│   └── validator.py           # Fail-closed post-redaction validator
│
├── reporting/
│   ├── __init__.py
│   ├── report_generator.py    # JSON & CSV audit report generators
│   └── metrics.py             # Precision, recall, F1, and structure retention
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Luhn validator, masking, IP validation, hashing
│   ├── generate_samples.py    # Synthetic business evidence generator
│   └── train_xgboost.py       # Train model and generate ML metrics JSON
│
├── tests/
│   ├── __init__.py
│   ├── test_extractors.py     # PDF, DOCX, PPTX, TXT unit tests
│   ├── test_detection.py      # Regex, NER, Context, Merging tests
│   ├── test_redaction.py      # Consistency and semantic replacement tests
│   └── test_pipeline.py       # End-to-end and fail-closed validation tests
│
├── sample_documents/          # Synthetic test documents (PDF, DOCX, PPTX, TXT)
├── uploads/                   # Local transient uploads
└── outputs/                   # Local sanitized outputs
```

---

## 12. Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Windows / macOS / Linux

### 1. Clone or Navigate to Project
```bash
cd "c:\Users\shris\Documents\OPTIV PROJECT"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy English Language Model (if not already installed)
```bash
python -m spacy download en_core_web_sm
```

### 4. (Optional) Generate Fresh Sample Evidence Documents
```bash
python utils/generate_samples.py
```

### 5. (Optional) Train the XGBoost Corroboration Module

The repository includes the trained artifact and metrics report. To retrain
from the provided Excel files:

```bash
python utils/train_xgboost.py ^
  --training-set "C:\path\Training_Set.xlsx" ^
  --testing-set "C:\path\Testing_Set.xlsx"
```

This writes the model and metrics report to `models/artifacts/`. The XGBoost
layer is optional at runtime; if its artifact is unavailable, the original
regex, context, and NER pipeline continues to operate.

---

## 13. Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

### Walkthrough of the User Experience:
1. **Upload Evidence Artifacts:** Select one or multiple business documents (PDF, DOCX, PPTX, TXT) using the file uploader, or click **"Load Sample Evidence"**.
2. **Execute Inspection:** Click **"Analyze Documents"**.
3. **Review Dashboard:**
   - **Overview Cards:** Real-time metrics for Documents Analyzed, Total PII Detected, High-Risk Findings, Files Sanitized, and Files Blocked.
   - **PII Breakdown:** Distribution across categories (PERSON, EMAIL, PHONE, EMPLOYEE_ID, etc.) and risk severity.
   - **Findings & Action Log:** Traceability table displaying severity, confidence, detection method, file, location, and masked entities.
   - **Document Deep-Dive:** Interactive tabs for inspecting Sanitized Output, Traceability Matrix, Validation Audit, Original Extracted Blocks, and the standalone **ML Metrics** report.
4. **Export Deliverables:**
   - Download sanitized text output (`.txt`).
   - Export comprehensive forensic audit report (`.json`).
   - Export findings traceability matrix (`.csv`).

---

## 14. Running Automated Tests

Run the complete test suite using `pytest`:
```bash
pytest -v
```

The test suite verifies:
- PDF extraction and page block retention
- Scanned PDF detection and OCR flagging
- DOCX headings, paragraphs, and tables extraction
- PPTX slides, text boxes, and tables extraction
- TXT multi-encoding extraction
- Email, phone, employee ID, government ID, and credit card regex detection
- spaCy NER person detection
- Context-sensitive rule detection
- Duplicate finding merging and confidence boosting
- Non-PII numerical values rejection (zero false alarms on revenue/dates)
- Semantic placeholder consistency across multiple mentions
- Reverse offset replacement integrity
- Fail-closed behavior on residual PII
- Fail-closed behavior on empty or corrupt files
- Full end-to-end pipeline execution

---

## 15. Current Limitations

- **OCR Engine:** V1 detects scanned PDF pages (`SCANNED_PAGE_REQUIRES_OCR`) and supports Tesseract when installed locally on the host system PATH. If Tesseract is not installed, it safely flags the page as requiring review without crashing.
- **Output Re-formatting:** V1 exports sanitized content as structured text retaining original structural headers, page labels, and slide sections. Direct re-compilation into native styled PDF/DOCX binary templates is planned for V2.
- **Ground Truth Metrics:** To maintain strict scientific honesty, benchmark precision/recall scores require ground-truth annotations and display *"Metrics unavailable — ground truth not provided"* when unannotated documents are uploaded.
- **XGBoost Benchmark Scope:** The current 100% XGBoost result is an entity-span classification result on the supplied annotated dataset, not an end-to-end document detection or redaction score. The test workbook is small and synthetic, and the data-generation process can produce regular patterns and overlap with training values. Independent real-world annotations are required for a reliable generalization estimate.
- **spaCy Model Availability:** If `en_core_web_sm` is not installed, the application safely continues with regex and context detection, but NER-specific tests and person-name coverage may be reduced. Install the model during setup for the full hybrid detector.

---

## 16. Future Enhancements

- **Native Format Export:** Re-rendering sanitized text back into formatted PDF, DOCX, and PPTX binaries.
- **Enterprise Vault Tokenization:** Reversible cryptographic tokenization vault for authorized human re-identification.
- **Advanced Context LLM Sidecar (Air-gapped):** Local quantized small language model (e.g. SLM) for advanced contextual ambiguity resolution.
- **Configurable Risk Policies:** Admin interface to define custom organization-specific PII thresholds and severity rules.
