"""PII Security Gateway - Pre-LLM Privacy & Document Sanitization Pipeline.

Enterprise Capstone Edition.
Vibrant white-background SaaS design system inspired by modern Figma security products.
Provides deterministic, explainable, and fail-closed privacy controls
for unstructured business documents prior to LLM context ingestion.
"""
import io
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models.schemas import (
    DocumentAnalysisResult,
    SecurityStatus,
    Severity,
    ConfidenceLevel,
    PIIType,
)
from pipeline import GatewayPipeline
from reporting.report_generator import ReportGenerator
from utils.helpers import calculate_file_hash

ML_METRICS_PATH = Path(__file__).resolve().parent / "models" / "artifacts" / "xgboost_ml_metrics.json"


def load_ml_metrics() -> Optional[Dict[str, Any]]:
    """Load the standalone XGBoost evaluation report when it is available."""
    if not ML_METRICS_PATH.exists():
        return None
    try:
        return json.loads(ML_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & VIBRANT WHITE DESIGN SYSTEM
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PII Security Gateway | Enterprise Privacy Gate",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VIBRANT_SAAS_CSS = """
<style>
    /* Design Tokens: Vibrant White-Background SaaS Theme */
    :root {
        --bg-app: #FFFFFF;
        --bg-surface: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-hover: #CBD5E1;
        
        --text-headline: #0F172A;
        --text-body: #1E293B;
        --text-secondary: #475569;
        --text-muted: #64748B;

        /* Vibrant Brand Palette */
        --brand-blue: #2563EB;
        --brand-indigo: #4F46E5;
        --brand-violet: #7C3AED;
        --brand-teal: #0D9488;
        --brand-cyan: #06B6D4;

        /* Security State Tones */
        --safe-green: #10B981;
        --safe-bg: #ECFDF5;
        --safe-border: #A7F3D0;
        --safe-text: #065F46;

        --blocked-red: #EF4444;
        --blocked-bg: #FEF2F2;
        --blocked-border: #FECACA;
        --blocked-text: #991B1B;

        --review-amber: #F59E0B;
        --review-bg: #FFFBEB;
        --review-border: #FDE68A;
        --review-text: #92400E;
    }

    /* Base Styling */
    .stApp {
        background-color: var(--bg-app) !important;
        color: var(--text-body) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-headline) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    /* Vibrant SaaS Header */
    .saas-header-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .saas-brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .saas-brand-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }
    .saas-brand-title {
        font-size: 20px;
        font-weight: 800;
        color: var(--text-headline);
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .saas-brand-badge {
        background: linear-gradient(135deg, #EFF6FF 0%, #EEF2FF 100%);
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .saas-brand-sub {
        font-size: 12.5px;
        color: var(--text-muted);
        margin-top: 2px;
    }
    .saas-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-sanitized {
        background-color: var(--safe-bg);
        border: 1px solid var(--safe-border);
        color: var(--safe-text);
    }
    .pill-blocked {
        background-color: var(--blocked-bg);
        border: 1px solid var(--blocked-border);
        color: var(--blocked-text);
    }
    .pill-review {
        background-color: var(--review-bg);
        border: 1px solid var(--review-border);
        color: var(--review-text);
    }
    .pill-staged {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
    }
    .pill-standby {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        color: var(--text-muted);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-sanitized { background-color: var(--safe-green); box-shadow: 0 0 6px var(--safe-green); }
    .dot-blocked { background-color: var(--blocked-red); box-shadow: 0 0 6px var(--blocked-red); }
    .dot-review { background-color: var(--review-amber); box-shadow: 0 0 6px var(--review-amber); }
    .dot-staged { background-color: #3B82F6; box-shadow: 0 0 6px #3B82F6; }
    .dot-standby { background-color: #94A3B8; }

    /* 5 Vibrant Metric Cards */
    .metric-card-vibrant {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .metric-card-vibrant:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.08);
        border-color: var(--border-hover);
    }
    .metric-stripe-indigo { border-top: 3px solid #4F46E5; }
    .metric-stripe-violet { border-top: 3px solid #7C3AED; }
    .metric-stripe-rose { border-top: 3px solid #EF4444; }
    .metric-stripe-emerald { border-top: 3px solid #10B981; }
    .metric-stripe-amber { border-top: 3px solid #F59E0B; }

    .metric-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .metric-label-txt {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-label-indigo { color: #4F46E5; }
    .metric-label-violet { color: #7C3AED; }
    .metric-label-rose { color: #EF4444; }
    .metric-label-emerald { color: #059669; }
    .metric-label-amber { color: #D97706; }

    .metric-icon-box {
        font-size: 15px;
    }
    .metric-main-val {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
        color: var(--text-headline);
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .metric-sub-txt {
        font-size: 11.5px;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Section Headers */
    .section-title-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .section-title-text {
        font-size: 13.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin: 0;
    }
    .section-desc-text {
        font-size: 12px;
        color: var(--text-muted);
        margin: 0;
    }

    /* Active Staged Artifact Card */
    .staged-artifact-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-left: 4px solid var(--brand-indigo);
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .staged-left-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .staged-badge-format {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        color: #4338CA;
        border: 1px solid #C7D2FE;
        font-size: 11px;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    .staged-name-text {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-headline);
    }
    .staged-meta-text {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 2px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* 5-Stage Visual Progression Stepper */
    .stepper-5stage-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 16px;
    }
    @media (max-width: 1024px) {
        .stepper-5stage-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    .step-box-cell {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: left;
        transition: all 0.2s ease;
    }
    .step-box-cell:hover {
        border-color: var(--border-hover);
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .step-tag-text {
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }
    .step-done-tag { color: #059669; }
    .step-title-text {
        font-size: 12.5px;
        font-weight: 700;
        color: var(--text-headline);
        margin-bottom: 2px;
    }
    .step-sub-text {
        font-size: 11px;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Executive Security Gate Verdict Banner */
    .verdict-banner-card {
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }
    .verdict-bg-sanitized {
        background-color: var(--safe-bg);
        border: 1px solid var(--safe-border);
    }
    .verdict-bg-blocked {
        background-color: var(--blocked-bg);
        border: 1px solid var(--blocked-border);
    }
    .verdict-bg-review {
        background-color: var(--review-bg);
        border: 1px solid var(--review-border);
    }
    .verdict-primary-col {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        max-width: 820px;
    }
    .verdict-icon-symbol {
        font-size: 24px;
        line-height: 1.1;
    }
    .verdict-title-headline {
        font-size: 15.5px;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .verdict-bg-sanitized .verdict-title-headline { color: var(--safe-text); }
    .verdict-bg-blocked .verdict-title-headline { color: var(--blocked-text); }
    .verdict-bg-review .verdict-title-headline { color: var(--review-text); }
    .verdict-body-desc {
        font-size: 13px;
        line-height: 1.5;
        color: var(--text-secondary);
    }
    .verdict-telemetry-strip {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .verdict-metric-pill {
        background-color: #FFFFFF;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 8px;
        padding: 6px 12px;
        text-align: center;
        min-width: 84px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .v-pill-lbl {
        font-size: 9.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }
    .v-pill-num {
        font-size: 15px;
        font-weight: 800;
        color: var(--text-headline);
    }

    /* Monospaced Document Preview */
    .preview-doc-box {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px 18px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12.5px;
        line-height: 1.7;
        color: var(--text-body);
        white-space: pre-wrap;
        max-height: 480px;
        overflow-y: auto;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.02);
    }

    /* Primary Action Button Polish: Vibrant Gradient */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-size: 13px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }

    /* Secondary / Download Buttons */
    div.stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: var(--text-headline) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        padding: 7px 16px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    div.stDownloadButton > button:hover {
        background-color: var(--bg-surface) !important;
        border-color: #C7D2FE !important;
        color: var(--brand-indigo) !important;
    }

    /* Streamlit Tabs Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-muted);
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: var(--brand-indigo) !important;
        border-bottom: 2px solid var(--brand-indigo) !important;
    }

    /* File Uploader Polishing */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--brand-indigo) !important;
        background-color: #FAFAFE !important;
    }

    /* Hide unnecessary Streamlit widgets */
    [data-testid="stFileChips"] > div > div:not(:last-child),
    [data-testid="stFileChips"] > *:first-child > *:not(:last-child),
    div[data-testid="stFileChips"] > div > div:not(:last-child),
    [data-testid="stFileUploader"] [data-testid="stFileChip"]:not(:last-child),
    [data-testid="stElementToolbar"],
    [data-testid="stDataFrameToolbar"],
    .stDataFrameToolbar,
    [data-testid="StyledFullScreenButton"],
    button[title="View fullscreen"],
    button[title="Download as CSV"],
    button[title="Search"],
    [data-testid="stHeaderActionElements"],
    a.header-anchor,
    .header-anchor {
        display: none !important;
        visibility: hidden !important;
    }

    /* Empty State Box */
    .saas-empty-container {
        background-color: #FFFFFF;
        border: 1px dashed var(--border-subtle);
        border-radius: 10px;
        padding: 36px 24px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .empty-symbol {
        font-size: 32px;
        color: var(--brand-indigo);
        margin-bottom: 10px;
    }
    .empty-lead {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-headline);
        margin-bottom: 4px;
    }
    .empty-detail {
        font-size: 13px;
        color: var(--text-muted);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }
</style>
"""
st.markdown(VIBRANT_SAAS_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. APPLICATION INITIALIZATION & HELPERS
# ------------------------------------------------------------------------------
@st.cache_resource
def get_pipeline() -> GatewayPipeline:
    """Initialize singleton GatewayPipeline instance."""
    return GatewayPipeline()


def format_bytes(size: int) -> str:
    """Format bytes into readable representation."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def apply_vibrant_theme(fig, height: int = 320):
    """Apply consistent vibrant SaaS theme to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=28, b=24),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#0F172A",
            size=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#475569"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        tickfont=dict(color="#475569", size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        tickfont=dict(color="#475569", size=11),
    )
    return fig


# Session State Management
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "active_view" not in st.session_state:
    st.session_state.active_view = "workspace"
if "last_active_file_signature" not in st.session_state:
    st.session_state.last_active_file_signature = None
if "active_sample_file" not in st.session_state:
    st.session_state.active_sample_file = None


# ------------------------------------------------------------------------------
# 3. HEADER RENDERING
# ------------------------------------------------------------------------------
def render_header(status_label: str, status_class: str, dot_class: str):
    """Render compact, vibrant SaaS header."""
    col_hdr, col_nav = st.columns([5, 2])

    with col_hdr:
        st.markdown(
            f"""
            <div class="saas-header-card">
                <div class="saas-brand-group">
                    <div class="saas-brand-icon">🛡️</div>
                    <div>
                        <div class="saas-brand-title">
                            PII Security Gateway
                            <span class="saas-brand-badge">Enterprise Edition</span>
                        </div>
                        <div class="saas-brand-sub">
                            Pre-LLM Document Privacy, Semantic Redaction & Fail-Closed Gate
                        </div>
                    </div>
                </div>
                <div>
                    <span class="saas-status-pill {status_class}">
                        <span class="status-dot {dot_class}"></span>
                        <span>{status_label}</span>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_nav:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        c_n1, c_n2, c_n3 = st.columns(3)
        with c_n1:
            if st.button("🔬 Gateway", use_container_width=True, help="Document Inspection Workspace"):
                st.session_state.active_view = "workspace"
                st.rerun()
        with c_n2:
            has_res = bool(st.session_state.get("analysis_results"))
            if st.button("📊 Telemetry", use_container_width=True, disabled=not has_res, help="View pictorial risk telemetry"):
                st.session_state.active_view = "telemetry"
                st.rerun()
        with c_n3:
            if st.button("📐 Architecture", use_container_width=True, help="Technical Specification & Policy Guide"):
                st.session_state.active_view = "architecture"
                st.rerun()


# ------------------------------------------------------------------------------
# 4. VIEW ROUTING: ARCHITECTURE vs TELEMETRY vs MAIN WORKSPACE
# ------------------------------------------------------------------------------
results: List[DocumentAnalysisResult] = st.session_state.get("analysis_results", [])
has_results = bool(results)

if st.session_state.get("active_view") == "architecture":
    # --------------------------------------------------------------------------
    # ARCHITECTURE & SPECIFICATION VIEW
    # --------------------------------------------------------------------------
    render_header("SPECIFICATION MODE", "pill-standby", "dot-standby")

    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid var(--border-subtle); border-left: 4px solid var(--brand-indigo); border-radius: 10px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); line-height: 1.65;">
            <h3 style="margin-top: 0; font-size: 18px; color: var(--brand-indigo); font-weight: 800;">
                Zero-Trust Pre-LLM Security Architecture & Policy Specification
            </h3>
            <p style="font-size: 13.5px; color: var(--text-secondary); margin-bottom: 14px;">
                The <b>PII Security Gateway</b> serves as an automated local cryptographic and privacy checkpoint situated between corporate unstructured document repositories and downstream Large Language Model (LLM) inference environments.
            </p>

            <h4 style="font-size: 14.5px; color: var(--text-headline); margin-top: 16px; margin-bottom: 8px;">1. Operational Threat Model</h4>
            <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary);">
                When employees upload business files (contracts, invoices, employee records, support tickets, incident reports) to LLMs, those files routinely contain confidential Personally Identifiable Information (PII) including full names, email addresses, phone numbers, employee badge IDs, credit cards, and government IDs. Transmitting unredacted documents breaches GDPR (Art. 25 & 32), HIPAA (§164.514), and the EU AI Act.
            </div>

            <h4 style="font-size: 14.5px; color: var(--text-headline); margin-top: 16px; margin-bottom: 8px;">2. Five-Stage Fail-Closed Pipeline</h4>
            <ol style="color: var(--text-secondary); font-size: 13px; padding-left: 20px; line-height: 1.75; margin-bottom: 16px;">
                <li><b>Stage 1 — Document Extraction:</b> Structural decomposition into immutable, ordered blocks (<code>page_1_block_2</code>) preserving layout boundaries across PDF, DOCX, PPTX, and TXT formats.</li>
                <li><b>Stage 2 — Multi-Engine Detection:</b> Corroborating hybrid scanner combining deterministic RFC-compliant regex, statistical spaCy NLP entity recognition with false-positive filtering, and proximity context rules.</li>
                <li><b>Stage 3 — Intelligent Risk Assessment:</b> Evaluates distinct entity volume, multi-attribute correlation hazard, and credential sensitivity to calculate an explainable Risk Score (0–100) and Risk Tier (LOW, MODERATE, HIGH, CRITICAL).</li>
                <li><b>Stage 4 — Consistent Semantic Redaction:</b> Generates deterministic surrogate keys (e.g. <code>[PERSON_001]</code>, <code>[EMAIL_001]</code>) rather than destructive blacking out, preserving co-reference resolution for AI reasoning.</li>
                <li><b>Stage 5 — Independent Secondary Validation:</b> An automated re-scan audits the sanitized text. If even a single residual PII entity is detected, the gateway fails closed (<b>BLOCKED ✕</b>). Downstream transmission is permitted strictly when zero residual PII is certified.</li>
            </ol>

            <div style="border-top: 1px solid var(--border-subtle); padding-top: 12px; font-size: 12.5px; color: var(--text-muted);">
                🔒 <b>Local Isolation Guarantee:</b> All parsing, extraction, tokenization, detection, and redaction procedures execute 100% locally within host memory. Zero document payload or telemetry is dispatched to external endpoints.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Return to Gateway Workspace", type="primary"):
        st.session_state.active_view = "workspace"
        st.rerun()

elif st.session_state.get("active_view") == "telemetry" and has_results:
    # --------------------------------------------------------------------------
    # VISUAL TELEMETRY VIEW
    # --------------------------------------------------------------------------
    render_header("TELEMETRY ACTIVE", "pill-sanitized", "dot-sanitized")

    active_doc = results[0]
    summary = ReportGenerator.get_summary_metrics(results)
    type_data = summary.get("type_breakdown", {})
    sev_data = summary.get("severity_breakdown", {})

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    c_t1, c_t2 = st.columns(2)

    with c_t1:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px 20px; box-shadow:0 1px 3px rgba(15,23,42,0.03);">
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:2px;">PII Classification Distribution</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:12px;">Entity frequency across detected privacy categories.</div>
            """,
            unsafe_allow_html=True,
        )
        if type_data:
            df_cat = pd.DataFrame(list(type_data.items()), columns=["Category", "Count"]).sort_values(by="Count", ascending=True)
            fig_cat = go.Figure(
                go.Bar(
                    x=df_cat["Count"],
                    y=df_cat["Category"],
                    orientation="h",
                    marker=dict(color="#4F46E5", line=dict(color="#3730A3", width=1)),
                    text=df_cat["Count"],
                    textposition="outside",
                )
            )
            fig_cat = apply_vibrant_theme(fig_cat, height=280)
            fig_cat.update_layout(xaxis_title="Entity Count", yaxis_title=None)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Zero PII detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_t2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px 20px; box-shadow:0 1px 3px rgba(15,23,42,0.03);">
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:2px;">Risk Severity Breakdown</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:12px;">Distribution across finding severity tiers.</div>
            """,
            unsafe_allow_html=True,
        )
        color_map = {"CRITICAL": "#DC2626", "HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
        sev_labels, sev_counts, sev_colors = [], [], []
        for sk in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if sk in sev_data and sev_data[sk] > 0:
                sev_labels.append(sk)
                sev_counts.append(sev_data[sk])
                sev_colors.append(color_map[sk])

        if sev_counts:
            fig_sev = go.Figure(
                data=[
                    go.Pie(
                        labels=sev_labels,
                        values=sev_counts,
                        hole=0.6,
                        marker=dict(colors=sev_colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_sev = apply_vibrant_theme(fig_sev, height=280)
            fig_sev.update_layout(showlegend=True)
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.success("Zero risk findings.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    if st.button("← Return to Gateway Workspace", type="primary"):
        st.session_state.active_view = "workspace"
        st.rerun()

else:
    # --------------------------------------------------------------------------
    # MAIN GATEWAY WORKSPACE
    # --------------------------------------------------------------------------
    active_sample = st.session_state.get("active_sample_file")

    # Determine Header Live State
    if has_results:
        active_doc = results[0]
        stat_val = active_doc.security_status.value if hasattr(active_doc.security_status, "value") else str(active_doc.security_status)
        if stat_val == "SANITIZED":
            hdr_lbl, hdr_cls, dot_cls = "GATEWAY: SANITIZED (PASS)", "pill-sanitized", "dot-sanitized"
        elif stat_val == "BLOCKED":
            hdr_lbl, hdr_cls, dot_cls = "GATEWAY: BLOCKED (FAIL-CLOSED)", "pill-blocked", "dot-blocked"
        else:
            hdr_lbl, hdr_cls, dot_cls = "GATEWAY: REVIEW REQUIRED", "pill-review", "dot-review"
    elif active_sample or st.session_state.get("last_active_file_signature"):
        hdr_lbl, hdr_cls, dot_cls = "ARTIFACT STAGED (READY)", "pill-staged", "dot-staged"
    else:
        hdr_lbl, hdr_cls, dot_cls = "STANDBY • READY FOR INGESTION", "pill-standby", "dot-standby"

    render_header(hdr_lbl, hdr_cls, dot_cls)

    # ==========================================================================
    # 1. FIVE VIBRANT METRIC CARDS (Calculated from Real Data)
    # ==========================================================================
    if has_results:
        summary_metrics = ReportGenerator.get_summary_metrics(results)
        m_docs = summary_metrics["total_documents"]
        m_pii = summary_metrics["total_pii_detected"]
        m_high = summary_metrics["high_risk_findings"]
        m_low = summary_metrics["low_risk_documents"]
        m_blocked_rev = summary_metrics["review_or_blocked"]
    else:
        m_docs = 0
        m_pii = 0
        m_high = 0
        m_low = 0
        m_blocked_rev = 0

    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)

    with c_m1:
        st.markdown(
            f"""
            <div class="metric-card-vibrant metric-stripe-indigo">
                <div class="metric-header-row">
                    <span class="metric-label-txt metric-label-indigo">Documents Processed</span>
                    <span class="metric-icon-box">📄</span>
                </div>
                <div class="metric-main-val">{m_docs}</div>
                <div class="metric-sub-txt">Active inspection scope</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m2:
        st.markdown(
            f"""
            <div class="metric-card-vibrant metric-stripe-violet">
                <div class="metric-header-row">
                    <span class="metric-label-txt metric-label-violet">Total Detected PII</span>
                    <span class="metric-icon-box">🔍</span>
                </div>
                <div class="metric-main-val" style="color: #7C3AED;">{m_pii}</div>
                <div class="metric-sub-txt">Flagged private entities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m3:
        st.markdown(
            f"""
            <div class="metric-card-vibrant metric-stripe-rose">
                <div class="metric-header-row">
                    <span class="metric-label-txt metric-label-rose">High-Risk Findings</span>
                    <span class="metric-icon-box">⚠️</span>
                </div>
                <div class="metric-main-val" style="color: {'#EF4444' if m_high > 0 else 'var(--text-headline)'};">{m_high}</div>
                <div class="metric-sub-txt">Critical & High risk findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m4:
        st.markdown(
            f"""
            <div class="metric-card-vibrant metric-stripe-emerald">
                <div class="metric-header-row">
                    <span class="metric-label-txt metric-label-emerald">Low-Risk Documents</span>
                    <span class="metric-icon-box">🛡️</span>
                </div>
                <div class="metric-main-val" style="color: #059669;">{m_low}</div>
                <div class="metric-sub-txt">Safe / low exposure profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m5:
        st.markdown(
            f"""
            <div class="metric-card-vibrant metric-stripe-amber">
                <div class="metric-header-row">
                    <span class="metric-label-txt metric-label-amber">Review or Blocked</span>
                    <span class="metric-icon-box">🛑</span>
                </div>
                <div class="metric-main-val" style="color: {'#D97706' if m_blocked_rev > 0 else '#059669'};">{m_blocked_rev}</div>
                <div class="metric-sub-txt">Gate holds & policy rejections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # 2. DOCUMENT INGESTION & STAGING WORKSPACE
    # ==========================================================================
    st.markdown(
        """
        <div class="section-title-wrap">
            <div>
                <h3 class="section-title-text">Document Ingestion & Staging</h3>
                <p class="section-desc-text">Upload business evidence artifact for pre-LLM PII detection and semantic sanitization</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up_main, col_up_sample = st.columns([3, 1])

    with col_up_main:
        uploaded_files = st.file_uploader(
            "Upload Document Artifact (Supported: PDF, DOCX, PPTX, TXT)",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True,
            key="saas_uploader",
            label_visibility="collapsed",
            help="Files are processed exclusively in host RAM. No data is stored externally.",
        )

    # Active file state handling
    active_file = None
    if uploaded_files:
        active_file = uploaded_files[-1]
        active_sig = f"{active_file.name}_{active_file.size}"
        if st.session_state.get("last_active_file_signature") != active_sig:
            st.session_state.analysis_results = []
            st.session_state.last_active_file_signature = active_sig
            st.session_state.active_sample_file = None
    else:
        if not st.session_state.get("active_sample_file"):
            if st.session_state.get("last_active_file_signature") is not None:
                st.session_state.analysis_results = []
                st.session_state.last_active_file_signature = None

    with col_up_sample:
        sample_options = {
            "Executive Briefing (PDF)": "Executive_Briefing.pdf",
            "Financial Org Pack (PPTX)": "Cadence_Financial_Group_Org_Pack.pptx",
            "HR Incident Report (DOCX)": "HR_Incident_Report.docx",
            "Customer Support Log (TXT)": "Customer_Support_Evidence.txt",
        }
        selected_sample_label = st.selectbox(
            "Benchmark Sample Corpus:",
            options=list(sample_options.keys()),
            key="saas_sample_select"
        )
        if st.button("📁 Stage Benchmark Artifact", use_container_width=True, help="Load test document from pre-built sample corpus"):
            sample_dir = Path(__file__).resolve().parent / "sample_documents"
            sample_file_path = sample_dir / sample_options[selected_sample_label]
            if sample_file_path.exists():
                st.session_state.active_sample_file = (sample_file_path.name, sample_file_path.read_bytes())
                st.session_state.last_active_file_signature = f"sample_{sample_file_path.name}"
                st.session_state.analysis_results = []
                st.rerun()

    # Active Staged Artifact Card
    staged_payload: Optional[Tuple[str, bytes]] = None
    staged_meta: Optional[Dict[str, str]] = None

    if active_file is not None:
        staged_payload = (active_file.name, active_file.getvalue())
        staged_meta = {
            "name": active_file.name,
            "ext": Path(active_file.name).suffix.replace(".", "").upper() or "FILE",
            "size": format_bytes(active_file.size),
            "source": "Uploaded Artifact",
            "hash": calculate_file_hash(active_file.getvalue()),
        }
    elif st.session_state.get("active_sample_file") is not None:
        s_name, s_bytes = st.session_state.active_sample_file
        staged_payload = (s_name, s_bytes)
        staged_meta = {
            "name": s_name,
            "ext": Path(s_name).suffix.replace(".", "").upper() or "FILE",
            "size": format_bytes(len(s_bytes)),
            "source": "Benchmark Sample Corpus",
            "hash": calculate_file_hash(s_bytes),
        }

    if staged_meta:
        col_staged_left, col_staged_cta = st.columns([4, 1.6])
        with col_staged_left:
            st.markdown(
                f"""
                <div class="staged-artifact-card">
                    <div class="staged-left-info">
                        <span class="staged-badge-format">{staged_meta['ext']}</span>
                        <div>
                            <div class="staged-name-text">{staged_meta['name']}</div>
                            <div class="staged-meta-text">
                                Source: {staged_meta['source']} • Size: {staged_meta['size']} • SHA-256: <code>{staged_meta['hash']}</code>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_staged_cta:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            run_inspection_btn = st.button("⚡ Execute Security Pipeline", type="primary", use_container_width=True)

        if run_inspection_btn and staged_payload:
            fname, fbytes = staged_payload
            prog_bar = st.progress(0, text=f"Initializing security pipeline for {fname}...")
            
            pipeline = get_pipeline()
            prog_bar.progress(20, text="1/5 Extracting structural document blocks...")
            time.sleep(0.04)
            prog_bar.progress(45, text="2/5 Executing hybrid tri-engine PII discovery...")
            time.sleep(0.04)
            prog_bar.progress(65, text="3/5 Computing multi-factor risk assessment...")
            time.sleep(0.04)
            prog_bar.progress(85, text="4/5 Applying consistent semantic surrogate redaction...")
            
            res = pipeline.process_document(file_bytes=fbytes, file_name=fname)
            prog_bar.progress(100, text="5/5 Post-redaction secondary validation verified.")
            time.sleep(0.04)
            prog_bar.empty()
            
            st.session_state.analysis_results = [res]
            st.rerun()

    else:
        st.markdown(
            """
            <div class="saas-empty-container">
                <div class="empty-symbol">📂</div>
                <div class="empty-lead">Ready for Document Security Ingestion</div>
                <div class="empty-detail">
                    Upload a business document (PDF, Word DOCX, Presentation PPTX, or Log TXT) or select a benchmark sample artifact to trigger the 5-stage Zero-Trust verification process.
                </div>
                <div style="margin-top: 12px; font-size: 11.5px; color: #94A3B8;">
                    🔒 All processing is executed locally in volatile RAM. No data egress to cloud LLMs or third parties.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # 3. FIVE-STAGE PIPELINE PROGRESSION & VERDICT
    # ==========================================================================
    if has_results:
        selected_doc = results[0]
        vr = selected_doc.validation_result
        doc_stat_val = selected_doc.security_status.value if hasattr(selected_doc.security_status, "value") else str(selected_doc.security_status)
        replacements_count = selected_doc.redaction_result.replacements_count if selected_doc.redaction_result else 0
        residual_count = len(vr.residual_findings) if vr else 0
        risk_data = selected_doc.risk_assessment or {}
        risk_tier = risk_data.get("risk_tier", "LOW")
        risk_score = risk_data.get("risk_score", 0)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title-wrap">
                <div>
                    <h3 class="section-title-text">Pipeline Progression & Security Gate Decision</h3>
                    <p class="section-desc-text">Sequential 5-stage verification outputs and authoritative security clearance</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5-Stage Stepper Grid
        s5_cls = "step-box-cell"
        s5_tag = "Stage 05 • Validation"
        s5_status = "0 Residual (Passed)" if (vr and vr.passed) else (f"{residual_count} Leaks (Failed)" if vr else "Pending")
        s5_style = "border-color: #A7F3D0; background-color: #F8FCF9;" if (vr and vr.passed) else ("border-color: #FECACA; background-color: #FEF2F2;" if vr else "")

        st.markdown(
            f"""
            <div class="stepper-5stage-grid">
                <div class="step-box-cell" style="border-color: #C7D2FE; background-color: #FAFAFF;">
                    <div class="step-tag-text step-done-tag">Stage 01 • Extraction</div>
                    <div class="step-title-text">Structural Parsing</div>
                    <div class="step-sub-text">{len(selected_doc.blocks)} blocks preserved</div>
                </div>
                <div class="step-box-cell" style="border-color: #DDD6FE; background-color: #FAF5FF;">
                    <div class="step-tag-text" style="color: #7C3AED;">Stage 02 • Detection</div>
                    <div class="step-title-text">Tri-Engine Discovery</div>
                    <div class="step-sub-text">{len(selected_doc.pii_findings)} entities flagged</div>
                </div>
                <div class="step-box-cell" style="border-color: #FED7AA; background-color: #FFFDF9;">
                    <div class="step-tag-text" style="color: #EA580C;">Stage 03 • Risk Assessment</div>
                    <div class="step-title-text">Risk Tier: {risk_tier}</div>
                    <div class="step-sub-text">Calculated Score: {risk_score}/100</div>
                </div>
                <div class="step-box-cell" style="border-color: #BFDBFE; background-color: #F8FAFF;">
                    <div class="step-tag-text" style="color: #2563EB;">Stage 04 • Redaction</div>
                    <div class="step-title-text">Semantic Replacement</div>
                    <div class="step-sub-text">{replacements_count} surrogate keys</div>
                </div>
                <div class="step-box-cell" style="{s5_style}">
                    <div class="step-tag-text" style="color: {'#059669' if (vr and vr.passed) else '#DC2626'};">{s5_tag}</div>
                    <div class="step-title-text">Secondary Gate</div>
                    <div class="step-sub-text">{s5_status}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Executive Gate Verdict Banner
        high_risk_count = sum(1 for f in selected_doc.pii_findings if (f.severity.value if hasattr(f.severity, "value") else str(f.severity)) in ("CRITICAL", "HIGH"))

        if doc_stat_val == SecurityStatus.SANITIZED.value:
            st.markdown(
                f"""
                <div class="verdict-banner-card verdict-bg-sanitized">
                    <div class="verdict-primary-col">
                        <div class="verdict-icon-symbol" style="color: #059669;">✓</div>
                        <div>
                            <div class="verdict-title-headline">SECURITY VERDICT: SANITIZED — APPROVED FOR DOWNSTREAM LLM INGESTION</div>
                            <div class="verdict-body-desc">
                                Post-redaction validation verified zero residual PII entities across all {len(selected_doc.blocks)} extracted blocks. Semantic surrogate keys maintain linguistic co-reference resolution for downstream AI reasoning while guaranteeing complete privacy.
                            </div>
                        </div>
                    </div>
                    <div class="verdict-telemetry-strip">
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Latency</div>
                            <div class="v-pill-num">{selected_doc.processing_time_ms:.1f}ms</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">PII Found</div>
                            <div class="v-pill-num">{len(selected_doc.pii_findings)}</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Risk Tier</div>
                            <div class="v-pill-num" style="color: #059669;">{risk_tier}</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Residual</div>
                            <div class="v-pill-num" style="color: #059669;">0</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif doc_stat_val == SecurityStatus.BLOCKED.value:
            st.markdown(
                f"""
                <div class="verdict-banner-card verdict-bg-blocked">
                    <div class="verdict-primary-col">
                        <div class="verdict-icon-symbol" style="color: #DC2626;">✕</div>
                        <div>
                            <div class="verdict-title-headline">SECURITY VERDICT: BLOCKED — FAIL-CLOSED ENFORCEMENT TRIGGERED</div>
                            <div class="verdict-body-desc">
                                Post-redaction re-scan detected residual PII entities or extraction anomaly. Downstream transmission to LLMs is halted immediately to prevent sensitive corporate data leakage.
                            </div>
                        </div>
                    </div>
                    <div class="verdict-telemetry-strip">
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Latency</div>
                            <div class="v-pill-num">{selected_doc.processing_time_ms:.1f}ms</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">PII Found</div>
                            <div class="v-pill-num">{len(selected_doc.pii_findings)}</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Risk Tier</div>
                            <div class="v-pill-num" style="color: #DC2626;">{risk_tier}</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Residual</div>
                            <div class="v-pill-num" style="color: #DC2626;">{residual_count}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-banner-card verdict-bg-review">
                    <div class="verdict-primary-col">
                        <div class="verdict-icon-symbol" style="color: #D97706;">⚠</div>
                        <div>
                            <div class="verdict-title-headline">SECURITY VERDICT: REVIEW REQUIRED — MANUAL CLEARANCE MANDATED</div>
                            <div class="verdict-body-desc">
                                Document contains low-confidence detections or unscanned embedded elements requiring compliance analyst verification prior to release.
                            </div>
                        </div>
                    </div>
                    <div class="verdict-telemetry-strip">
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Latency</div>
                            <div class="v-pill-num">{selected_doc.processing_time_ms:.1f}ms</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">PII Found</div>
                            <div class="v-pill-num">{len(selected_doc.pii_findings)}</div>
                        </div>
                        <div class="verdict-metric-pill">
                            <div class="v-pill-lbl">Risk Tier</div>
                            <div class="v-pill-num" style="color: #D97706;">{risk_tier}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ======================================================================
        # 4. TABBED INVESTIGATION & TRIAGE CONSOLE
        # ======================================================================
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title-wrap">
                <div>
                    <h3 class="section-title-text">Security Operations & Analysis Console</h3>
                    <p class="section-desc-text">Detailed entity triage, intelligent risk assessment, sanitized artifact preview, and traceability matrix</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_findings, tab_risk, tab_sanitized, tab_validation, tab_traceability, tab_blocks, tab_exports, tab_ml_metrics = st.tabs([
            "🔍 PII Findings Log",
            "⚖️ Intelligent Risk Assessment",
            "📄 Sanitized Output & Surrogate Registry",
            "🛡️ Post-Redaction Validation Audit",
            "🧭 Structural Traceability Matrix",
            "📑 Raw Extracted Blocks",
            "📋 Compliance Export (JSON/CSV)",
            "📈 ML Metrics",
        ])

        # TAB 1: PII FINDINGS LOG & TRIAGE
        with tab_findings:
            col_f_sev, col_f_cat, col_unmask = st.columns([1.5, 1.5, 2])
            with col_f_sev:
                sev_choice = st.selectbox(
                    "Filter Severity:",
                    ["All Severities", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    key="vibrant_sev_filter"
                )
            with col_f_cat:
                all_cats = sorted(list({f.pii_type.value if hasattr(f.pii_type, "value") else str(f.pii_type) for f in selected_doc.pii_findings}))
                cat_choice = st.selectbox(
                    "Filter Classification:",
                    ["All Classifications"] + all_cats,
                    key="vibrant_cat_filter"
                )
            with col_unmask:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                reveal_raw = st.checkbox(
                    "Analyst Clearance: Reveal Unmasked Values",
                    value=False,
                    help="Controlled access for authorized SOC analysts to inspect plaintext values."
                )

            filtered_rows = []
            for f in selected_doc.pii_findings:
                f_sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
                f_type = f.pii_type.value if hasattr(f.pii_type, "value") else str(f.pii_type)

                if sev_choice != "All Severities" and f_sev != sev_choice:
                    continue
                if cat_choice != "All Classifications" and f_type != cat_choice:
                    continue

                display_val = f.text if reveal_raw else f.masked_text
                filtered_rows.append({
                    "Severity": f_sev,
                    "Classification": f_type,
                    "Confidence": f"{f.confidence * 100:.1f}% ({f.confidence_level.value})",
                    "Method": " + ".join(f.detection_method),
                    "Location": f.location,
                    "Entity Representation": display_val,
                    "Risk Rationale": f.explanation,
                })

            if filtered_rows:
                st.dataframe(pd.DataFrame(filtered_rows), use_container_width=True, hide_index=True)
                st.markdown(f"<div style='font-size: 11.5px; color: var(--text-muted); margin-top: 4px;'>Displaying <b>{len(filtered_rows)}</b> of <b>{len(selected_doc.pii_findings)}</b> verified findings.</div>", unsafe_allow_html=True)
            else:
                if selected_doc.pii_findings:
                    st.info("No findings match the selected filter criteria.")
                else:
                    st.success("Zero PII entities detected across this document.")

        # TAB 2: INTELLIGENT RISK ASSESSMENT
        with tab_risk:
            st.markdown("<div style='font-size: 13.5px; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Explainable Document Privacy Risk Breakdown:</div>", unsafe_allow_html=True)
            
            c_rk1, c_rk2, c_rk3, c_rk4 = st.columns(4)
            with c_rk1:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; text-align:center;">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Overall Risk Tier</div>
                        <div style="font-size:22px; font-weight:800; color:{'#DC2626' if risk_tier in ('CRITICAL', 'HIGH') else ('#D97706' if risk_tier == 'MODERATE' else '#059669')}; margin-top:2px;">{risk_tier}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_rk2:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; text-align:center;">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Calculated Score</div>
                        <div style="font-size:22px; font-weight:800; color:#4F46E5; margin-top:2px;">{risk_score} / 100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_rk3:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; text-align:center;">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Distinct Entities</div>
                        <div style="font-size:22px; font-weight:800; color:#0F172A; margin-top:2px;">{risk_data.get('distinct_entities_count', 0)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_rk4:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; text-align:center;">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Regulated Credentials</div>
                        <div style="font-size:22px; font-weight:800; color:{'#DC2626' if risk_data.get('critical_identifiers_present') else '#059669'}; margin-top:2px;">{'YES' if risk_data.get('critical_identifiers_present') else 'NO'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12.5px; font-weight: 700; color: #334155;'>Risk Factors & Justification:</div>", unsafe_allow_html=True)
            for factor in risk_data.get("risk_factors", []):
                st.markdown(f"- **Evidence**: {factor}")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown(f"**Policy Recommendation**: {risk_data.get('recommendation', 'N/A')}")

        # TAB 3: SANITIZED ARTIFACT & SURROGATE KEYS
        with tab_sanitized:
            if selected_doc.redaction_result and selected_doc.redaction_result.redacted_full_text:
                col_prev_txt, col_prev_dict = st.columns([3, 2])

                with col_prev_txt:
                    st.markdown("<div style='font-size: 12.5px; font-weight: 700; color: #0F172A; margin-bottom: 6px;'>Sanitized Document Content (Semantic Surrogate Tokens Preserved):</div>", unsafe_allow_html=True)
                    st.markdown(f'<div class="preview-doc-box">{selected_doc.redaction_result.redacted_full_text}</div>', unsafe_allow_html=True)

                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    if doc_stat_val == SecurityStatus.SANITIZED.value:
                        st.download_button(
                            label=f"⬇ Download Sanitized Document ({Path(selected_doc.file_name).stem}_sanitized.txt)",
                            data=selected_doc.redaction_result.redacted_full_text,
                            file_name=f"{Path(selected_doc.file_name).stem}_sanitized.txt",
                            mime="text/plain",
                        )
                    else:
                        st.warning("⚠️ Sanitized artifact download is locked because post-redaction validation did not certify clean status.")

                with col_prev_dict:
                    st.markdown("<div style='font-size: 12.5px; font-weight: 700; color: #0F172A; margin-bottom: 6px;'>Deterministic Surrogate Key Mapping Registry:</div>", unsafe_allow_html=True)
                    mapping_entries = []
                    for raw_key, tag in selected_doc.redaction_result.entity_mappings.items():
                        mapping_entries.append({
                            "Surrogate Key": tag,
                            "Masked Entity": raw_key if reveal_raw else f"•••••••• ({len(raw_key)} chars)",
                            "Co-Reference Standard": "Deterministic 1:1 Tag",
                        })
                    if mapping_entries:
                        st.dataframe(pd.DataFrame(mapping_entries), use_container_width=True, hide_index=True)
                        st.markdown("<div style='font-size: 11px; color: var(--text-muted); margin-top: 4px;'>Recurring mentions of the same identity map to identical surrogate tokens across pages.</div>", unsafe_allow_html=True)
                    else:
                        st.info("No surrogate keys generated.")
            else:
                st.info("No redaction outputs available.")

        # TAB 4: POST-REDACTION VALIDATION AUDIT
        with tab_validation:
            if vr:
                stat_badge_color = "#059669" if vr.passed else "#DC2626"
                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 16px 20px; margin-bottom: 14px;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                            <tr style="border-bottom: 1px solid var(--border-subtle);">
                                <td style="padding: 8px 0; color: var(--text-muted); font-weight: 600;">Secondary Validation Gate:</td>
                                <td style="padding: 8px 0; font-weight: 700; color: {stat_badge_color};">{'PASSED (ZERO RESIDUAL PII)' if vr.passed else 'FAILED (RESIDUAL DETECTIONS FOUND)'}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-subtle);">
                                <td style="padding: 8px 0; color: var(--text-muted); font-weight: 600;">Residual PII Count:</td>
                                <td style="padding: 8px 0; font-weight: 700; color: var(--text-headline);">{len(vr.residual_findings)} entities</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-subtle);">
                                <td style="padding: 8px 0; color: var(--text-muted); font-weight: 600;">Fail-Closed Security Verdict:</td>
                                <td style="padding: 8px 0; font-weight: 700; color: var(--text-headline);">{vr.status.value}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: var(--text-muted); font-weight: 600;">Enforced Security Rule:</td>
                                <td style="padding: 8px 0; color: var(--text-secondary);">{vr.validation_message}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if vr.residual_findings:
                    st.error(f"✕ Quarantine Alert: {len(vr.residual_findings)} residual PII entities detected during secondary audit:")
                    residual_rows = []
                    for rf in vr.residual_findings:
                        residual_rows.append({
                            "Location": rf.location,
                            "Classification": rf.pii_type.value if hasattr(rf.pii_type, "value") else str(rf.pii_type),
                            "Residual Plaintext": rf.text,
                            "Confidence": f"{rf.confidence * 100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(residual_rows), use_container_width=True, hide_index=True)
                else:
                    st.success("✓ Secondary re-scan certified 0 residual PII leaks. All original structural boundaries maintained.")
            else:
                st.info("Validation audit logs unavailable.")

        # TAB 5: STRUCTURAL TRACEABILITY MATRIX
        with tab_traceability:
            if selected_doc.pii_findings and selected_doc.redaction_result:
                traceability_rows = []
                for f in selected_doc.pii_findings:
                    assigned_tag = selected_doc.redaction_result.entity_mappings.get(f.text.strip(), "N/A")
                    traceability_rows.append({
                        "Source Location": f.location,
                        "PII Classification": f.pii_type.value if hasattr(f.pii_type, "value") else str(f.pii_type),
                        "Confidence Score": f"{f.confidence * 100:.1f}% ({f.confidence_level.value})",
                        "Corroborating Engine": " + ".join(f.detection_method),
                        "Surrogate Token Assigned": assigned_tag,
                        "Risk Justification": f.explanation,
                    })
                st.dataframe(pd.DataFrame(traceability_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Traceability matrix records unavailable.")

        # TAB 6: RAW EXTRACTED BLOCKS
        with tab_blocks:
            st.markdown(f"<div style='font-size: 12px; color: var(--text-muted); margin-bottom: 10px;'>Preserved structural blocks for <b>{selected_doc.file_name}</b> (Retained in volatile local memory):</div>", unsafe_allow_html=True)
            for b in selected_doc.blocks:
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid #6366F1; border-radius:4px; padding:10px 14px; margin-bottom:8px;">
                        <div style="font-size:11px; font-weight:600; color:#64748B; margin-bottom:4px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">Coordinate: {b.location} | Type: {b.block_type} | Index: {b.order}</div>
                        <div style="font-size:12.5px; color:#1E293B; white-space:pre-wrap; line-height:1.5;">{b.text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # TAB 7: COMPLIANCE EXPORTS
        with tab_exports:
            st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Export SIEM-Ready Audit Telemetry & Forensic Traceability Matrix:</div>", unsafe_allow_html=True)
            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                json_data = ReportGenerator.generate_json_report(results)
                st.download_button(
                    label="⬇ Export Security Audit Dossier (JSON)",
                    data=json_data,
                    file_name="pii_security_gateway_audit_dossier.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with c_exp2:
                csv_data = ReportGenerator.generate_csv_findings(results)
                st.download_button(
                    label="⬇ Export Findings Traceability Matrix (CSV)",
                    data=csv_data,
                    file_name="pii_findings_traceability_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # TAB 8: STANDALONE ML METRICS
        with tab_ml_metrics:
            ml_metrics = load_ml_metrics()
            st.markdown(
                "<div style='font-size: 13px; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>XGBoost PII Classification Metrics</div>",
                unsafe_allow_html=True,
            )
            if not ml_metrics:
                st.info("ML metrics are unavailable. Train the optional XGBoost model to generate the metrics file.")
            else:
                report = ml_metrics.get("testing_report", {})
                overall = report.get("weighted avg", {})
                metric_cols = st.columns(4)
                metric_values = [
                    ("Accuracy", report.get("accuracy", 0.0)),
                    ("Precision", overall.get("precision", 0.0)),
                    ("Recall", overall.get("recall", 0.0)),
                    ("F1 Score", overall.get("f1-score", 0.0)),
                ]
                for col, (label, value) in zip(metric_cols, metric_values):
                    with col:
                        st.metric(label, f"{float(value) * 100:.1f}%")

                class_rows = []
                for label, values in report.items():
                    if isinstance(values, dict) and "precision" in values:
                        class_rows.append({
                            "Class": label,
                            "Precision": f"{values['precision'] * 100:.1f}%",
                            "Recall": f"{values['recall'] * 100:.1f}%",
                            "F1 Score": f"{values['f1-score'] * 100:.1f}%",
                            "Test Support": int(values.get("support", 0)),
                        })
                if class_rows:
                    st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"Training examples: {ml_metrics.get('training_examples', 'N/A')} | "
                    f"Classes: {len(ml_metrics.get('classes', []))}. "
                    "Metrics are specific to the supplied annotated dataset."
                )
                st.download_button(
                    label="⬇ Download ML Metrics (JSON)",
                    data=json.dumps(ml_metrics, indent=2),
                    file_name="xgboost_ml_metrics.json",
                    mime="application/json",
                    use_container_width=True,
                )
