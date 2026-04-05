from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if LOCAL_SITE_PACKAGES.exists():
    local_site_packages_str = str(LOCAL_SITE_PACKAGES)
    if local_site_packages_str not in sys.path:
        sys.path.insert(0, local_site_packages_str)

import pandas as pd
import streamlit as st

from data_ingestion.data_logger import ensure_log_files, log_chat_event, log_quiz_attempt, log_user_activity
from database.database_manager import (
    add_chat_message,
    authenticate_user,
    create_chat_session,
    database_status,
    export_table,
    get_document,
    get_documents_df,
    get_events_df,
    get_question_bank_df,
    get_quiz_df,
    get_study_df,
    get_summary_df,
    get_user,
    get_user_performance_summary,
    get_users_df,
    get_or_create_document,
    initialize_database_schema,
    list_chat_messages,
    list_chat_sessions,
    log_event,
    log_study_session,
    persist_pipeline_report,
    rate_chat_message,
    register_user,
    save_quiz_result,
    update_question_quality,
)
from learnmate_ai.config import get_config
from learnmate_ai.storage import ensure_data_directories, save_uploaded_file, timestamped_name
from modules.utils import strip_page_markers


DOC_PATH = "data/latest_doc.txt"
SUMMARY_MODES = ["brief", "detailed", "bullet_points"]
CHAT_ANSWER_MODES = {
    "Explain Like Teacher": "teacher",
    "Short Answer": "short",
    "Step-by-Step": "step_by_step",
}


def inject_theme() -> None:
    # Google Fonts — must be a separate markdown call
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* ── Base & Typography ────────────────────────────────────────── */
        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        h1, h2, h3, h4, h5, h6, p, label, li, a {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .stApp {
            background: linear-gradient(160deg, #f8faff 0%, #eef2ff 40%, #f0fdf4 100%);
            color: #1e293b;
        }
        h1, h2, h3, h4, h5, h6, p, li, label {
            color: #1e293b;
        }

        /* ── Header ──────────────────────────────────────────────────── */
        [data-testid="stHeader"] {
            background: rgba(248, 250, 255, 0.9);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(99, 102, 241, 0.08);
        }

        /* ── Sidebar — BRIGHT & CLEAN ────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f0f0ff 40%, #e8e0ff 100%) !important;
            border-right: 2px solid rgba(139, 92, 246, 0.15);
            box-shadow: 4px 0 30px rgba(139, 92, 246, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #3730a3 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #312e81 !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        [data-testid="stSidebar"] p {
            color: #4338ca !important;
            line-height: 1.5 !important;
        }
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption * {
            color: #7c3aed !important;
            font-size: 0.8rem !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(139, 92, 246, 0.2) !important;
            margin: 12px 0 !important;
        }

        /* Sidebar navigation radio buttons */
        section[data-testid="stSidebar"] .stRadio > label {
            color: #312e81 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            margin-bottom: 6px !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 4px !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.85) !important;
            border: 1.5px solid rgba(139, 92, 246, 0.2) !important;
            border-radius: 12px !important;
            padding: 10px 16px !important;
            margin-bottom: 5px !important;
            transition: all 0.25s ease !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            color: #3730a3 !important;
            display: flex !important;
            align-items: center !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
            color: #3730a3 !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: rgba(139, 92, 246, 0.1) !important;
            border-color: #8b5cf6 !important;
            transform: translateX(3px);
            box-shadow: 0 2px 12px rgba(139, 92, 246, 0.15) !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
            background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
            border-color: transparent !important;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3) !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] *,
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] * {
            color: #ffffff !important;
        }

        /* Sidebar select boxes */
        section[data-testid="stSidebar"] .stSelectbox label {
            color: #312e81 !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1.5px solid rgba(139, 92, 246, 0.2) !important;
            border-radius: 10px !important;
            color: #3730a3 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div * {
            color: #3730a3 !important;
        }

        /* Sidebar text inputs */
        section[data-testid="stSidebar"] .stTextArea label,
        section[data-testid="stSidebar"] .stTextInput label {
            color: #312e81 !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] .stTextArea textarea,
        section[data-testid="stSidebar"] .stTextInput input {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1.5px solid rgba(139, 92, 246, 0.2) !important;
            border-radius: 10px !important;
            color: #3730a3 !important;
        }
        section[data-testid="stSidebar"] .stTextArea textarea:focus,
        section[data-testid="stSidebar"] .stTextInput input:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
        }
        section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
        section[data-testid="stSidebar"] .stTextInput input::placeholder {
            color: #a78bfa !important;
        }

        /* Sidebar file uploader */
        section[data-testid="stSidebar"] .stFileUploader label {
            color: #312e81 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            margin-bottom: 6px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
            margin-bottom: 12px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 2px dashed rgba(139, 92, 246, 0.25) !important;
            border-radius: 14px !important;
            padding: 16px 12px !important;
            min-height: 80px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            padding: 8px 4px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] span {
            color: #6d28d9 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-align: center !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small {
            color: #8b5cf6 !important;
            font-size: 0.72rem !important;
            text-align: center !important;
            display: block !important;
            line-height: 1.3 !important;
            margin-top: 2px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small * {
            color: #8b5cf6 !important;
            font-size: 0.72rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * {
            color: #7c3aed !important;
            fill: #7c3aed !important;
            opacity: 1 !important;
        }
        /* Browse button inside uploader */
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 6px 18px !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            margin-top: 4px !important;
            cursor: pointer !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        /* Sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.25) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        section[data-testid="stSidebar"] .stButton > button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.35) !important;
            filter: brightness(1.08) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stBaseButton-secondary"],
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
            background: rgba(255, 255, 255, 0.85) !important;
            color: #6d28d9 !important;
            border: 1.5px solid rgba(139, 92, 246, 0.25) !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stBaseButton-secondary"] *,
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] * {
            color: #6d28d9 !important;
            fill: #6d28d9 !important;
        }

        /* Sidebar markdown content — prevent overlap */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            overflow-wrap: break-word !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 4px !important;
            line-height: 1.5 !important;
            font-size: 0.88rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.3rem !important;
            margin-bottom: 2px !important;
            margin-top: 4px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            font-size: 1.05rem !important;
            margin-top: 12px !important;
            margin-bottom: 6px !important;
        }

        /* Sidebar scrollbar */
        section[data-testid="stSidebar"]::-webkit-scrollbar { width: 5px; }
        section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: transparent; }
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 5px; }

        /* ── Main-area Buttons ───────────────────────────────────────── */
        .block-container .stButton > button,
        .block-container button[kind="secondary"],
        .block-container button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1.6rem !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25), 0 1px 3px rgba(0,0,0,0.08) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .block-container .stButton > button *,
        .block-container button[kind="secondary"] *,
        .block-container button[kind="primary"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        .block-container .stButton > button:hover,
        .block-container button[kind="secondary"]:hover,
        .block-container button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35), 0 2px 6px rgba(0,0,0,0.1) !important;
            filter: brightness(1.05) !important;
        }
        .block-container .stButton > button:active, .stDownloadButton > button:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
        }

        /* ── Form Inputs (main area) ─────────────────────────────────── */
        .stSelectbox label, .stRadio label, .stTextArea label,
        .stTextInput label, .stSlider label, .stFileUploader label {
            color: #1e293b !important;
            font-weight: 600;
            font-size: 0.875rem;
        }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            background: #ffffff !important;
            color: #1e293b !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 10px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #94a3b8 !important;
        }

        /* ── Radio Groups (main area) ────────────────────────────────── */
        div[role="radiogroup"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            color: #1e293b !important;
            opacity: 1 !important;
        }
        div[data-testid="stRadio"] p {
            color: #1e293b !important;
        }
        .block-container .stRadio div[role="radiogroup"] label {
            background: #ffffff;
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            padding: 8px 14px;
            margin-bottom: 4px;
            transition: all 0.2s ease;
        }
        .block-container .stRadio div[role="radiogroup"] label:hover {
            background: #eef2ff;
            border-color: #8b5cf6;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.08);
        }

        /* ── File Uploader (main area) ───────────────────────────────── */
        div[data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border: 2px dashed #cbd5e1 !important;
            border-radius: 14px !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
        }
        div[data-testid="stFileUploaderDropzone"] * {
            color: #64748b !important;
            fill: #64748b !important;
            opacity: 1 !important;
        }
        div[data-testid="stBaseButton-secondary"],
        button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {
            background: #ffffff !important;
            color: #1e293b !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 10px !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stBaseButton-secondary"]:hover,
        button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background: #f8fafc !important;
            border-color: #cbd5e1 !important;
        }
        div[data-testid="stBaseButton-secondary"] *,
        button[kind="secondary"] *,
        button[data-testid="stBaseButton-secondary"] * {
            color: #1e293b !important;
            fill: #1e293b !important;
        }

        /* ── Markdown Text ───────────────────────────────────────────── */
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] span {
            color: #1e293b !important;
            line-height: 1.65;
        }
        .stMarkdown h1 { color: #312e81; font-weight: 800; }
        .stMarkdown h2 { color: #312e81; font-weight: 700; }
        .stMarkdown h3 { color: #3730a3; font-weight: 700; font-size: 1.2rem; }
        .stMarkdown h4 { color: #4338ca; font-weight: 600; }

        /* ── Alerts ──────────────────────────────────────────────────── */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 12px !important;
            border: none !important;
            font-weight: 500;
        }
        .stSuccess {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0) !important;
            color: #166534 !important;
        }
        .stInfo {
            background: linear-gradient(135deg, #eef2ff, #c7d2fe) !important;
            color: #3730a3 !important;
        }
        .stWarning {
            background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
            color: #92400e !important;
        }
        .stError {
            background: linear-gradient(135deg, #fee2e2, #fecaca) !important;
            color: #991b1b !important;
        }

        /* ── Metric Cards ────────────────────────────────────────────── */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1.5px solid rgba(139, 92, 246, 0.1);
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.06), 0 1px 3px rgba(0,0,0,0.03);
            transition: all 0.25s ease;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.12), 0 2px 6px rgba(0,0,0,0.04);
            transform: translateY(-3px);
            border-color: rgba(139, 92, 246, 0.25);
        }
        div[data-testid="stMetric"] label {
            color: #6d28d9 !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 700 !important;
            font-size: 1.6rem !important;
            color: #312e81 !important;
        }

        /* ── DataFrames ──────────────────────────────────────────────── */
        div[data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(6px);
            border-radius: 14px;
            padding: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            border: 1.5px solid rgba(139, 92, 246, 0.1);
        }

        /* ── Charts ──────────────────────────────────────────────────── */
        div[data-testid="stVegaLiteChart"] {
            background: rgba(255, 255, 255, 0.75);
            border-radius: 14px;
            padding: 8px;
            border: 1.5px solid rgba(139, 92, 246, 0.08);
        }

        /* ── Tabs ────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(238, 242, 255, 0.8);
            border-radius: 12px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 600;
            color: #64748b !important;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #312e81 !important;
            background: rgba(255,255,255,0.5);
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: #6d28d9 !important;
            box-shadow: 0 1px 4px rgba(99, 102, 241, 0.1);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* ── Expanders ───────────────────────────────────────────────── */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.6) !important;
            border-radius: 12px !important;
            font-weight: 600;
            color: #1e293b !important;
            transition: background 0.2s ease;
        }
        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.85) !important;
        }
        details[data-testid="stExpander"] {
            border: 1.5px solid rgba(139, 92, 246, 0.1) !important;
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.5);
        }

        /* ── Progress Bar ────────────────────────────────────────────── */
        .stProgress > div > div {
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa) !important;
            border-radius: 20px;
        }
        .stProgress > div {
            background: #e2e8f0 !important;
            border-radius: 20px;
        }

        /* ── Layout Container ────────────────────────────────────────── */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* ── Custom scrollbar ────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: #a78bfa; }

        /* ── Sidebar Nav ─────────────────────────────────────────────── */
        [data-testid="stSidebarNav"] {
            color: #3730a3 !important;
        }

        /* ── Form submit button matching ─────────────────────────────── */
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
        }
        .stFormSubmitButton > button * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "active_user_id": None,
        "active_user_name": "",
        "active_user_email": "",
        "dataset_df": None,
        "dataset_name": None,
        "dataset_raw_path": None,
        "quiz_package": None,
        "pipeline_report": None,
        "pipeline_summary": None,
        "last_quiz_score": None,
        "current_document_topic": "general_document",
        "current_document_name": None,
        "current_document_id": None,
        "uploaded_doc_ids": [],
        "uploaded_doc_names": [],
        "current_page": "Dashboard",
        "summary_result": None,
        "chat_session_id": None,
        "last_assistant_message_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Automatically restore session if query param auth_token is present
    if not st.session_state.get("authenticated") and "auth_token" in st.query_params:
        try:
            stored_id = int(st.query_params["auth_token"])
            user = get_user(stored_id)
            if user:
                st.session_state.authenticated = True
                st.session_state.active_user_id = user["id"]
                st.session_state.active_user_name = user["full_name"]
                st.session_state.active_user_email = user["email"]
        except Exception:
            pass

    # Automatically restore last active document for this user to prevent upload wipe-outs
    if st.session_state.get("authenticated") and not st.session_state.get("current_document_id"):
        try:
            from database.database_manager import get_documents_df
            df = get_documents_df()
            user_docs = df[df["user_id"] == st.session_state.active_user_id]
            if not user_docs.empty:
                latest = user_docs.iloc[0]
                doc_id = int(latest["id"])
                
                from database.database_manager import get_document
                doc_data = get_document(doc_id)
                if doc_data:
                    st.session_state.current_document_id = doc_id
                    st.session_state.current_document_name = latest["filename"]
                    st.session_state.current_document_topic = latest["topic"]
                    st.session_state.uploaded_doc_ids = [doc_id]
                    st.session_state.uploaded_doc_names = [latest["filename"]]
                    
                    with open(DOC_PATH, "w", encoding="utf-8") as f:
                        f.write(doc_data["text_content"])
        except Exception:
            pass


def current_user_id() -> int | None:
    return st.session_state.get("active_user_id")


def current_document_topic() -> str:
    return st.session_state.get("current_document_topic") or "general_document"


def current_subject() -> str:
    return st.session_state.get("current_document_name") or "Uploaded Document"


def current_document_id() -> int | None:
    return st.session_state.get("current_document_id")


def load_document_text() -> str | None:
    if not os.path.exists(DOC_PATH):
        return None
    with open(DOC_PATH, "r", encoding="utf-8") as file:
        return file.read()


def estimate_study_minutes(text: str) -> int:
    words = len(text.split())
    return max(5, min(90, words // 180))


# ── Cached data loaders (avoid redundant SQL on every re-render) ─────
@st.cache_data(ttl=30, show_spinner=False)
def _cached_quiz_df(_config):
    return get_quiz_df(_config)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_study_df(_config):
    return get_study_df(_config)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_documents_df(_config):
    return get_documents_df(_config)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_summary_df(_config):
    return get_summary_df(_config)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_events_df(_config, limit=200):
    return get_events_df(limit=limit, config=_config)


def logout() -> None:
    for key, value in {
        "authenticated": False,
        "active_user_id": None,
        "active_user_name": "",
        "active_user_email": "",
        "quiz_package": None,
        "last_quiz_score": None,
        "summary_result": None,
        "chat_session_id": None,
        "last_assistant_message_id": None,
        "uploaded_doc_ids": [],
        "uploaded_doc_names": [],
        "current_document_id": None,
        "current_document_name": None,
        "current_document_topic": "general_document",
    }.items():
        st.session_state[key] = value
    if "auth_token" in st.query_params:
        del st.query_params["auth_token"]
    st.rerun()


def render_auth_page(config) -> None:
    # Centered branded header
    st.markdown(
        """<div style="text-align: center; padding: 40px 0 10px 0;">
        <h1 style="font-size: 2.4rem; font-weight: 800; margin: 0;
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;">LearnMate AI</h1>
        <p style="color: #6d28d9; font-size: 1rem; margin-top: 8px;">Your intelligent study companion</p>
        </div>""",
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.4, 1])
    with center:
        st.markdown(
            """<div style="background: rgba(255,255,255,0.85); backdrop-filter: blur(10px);
            border: 1px solid rgba(226,232,240,0.6); border-radius: 20px;
            padding: 8px 0 4px 0; margin-bottom: 12px;"></div>""",
            unsafe_allow_html=True,
        )
        tabs = st.tabs(["Sign In", "Sign Up"])
        with tabs[0]:
            with st.form("login_form", clear_on_submit=True):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_clicked = st.form_submit_button("Sign In", use_container_width=True)
            if login_clicked:
                try:
                    result = authenticate_user(email, password, config)
                    st.session_state.authenticated = True
                    st.session_state.active_user_id = result["user_id"]
                    st.session_state.active_user_name = result["full_name"]
                    st.session_state.active_user_email = result["email"]
                    st.query_params["auth_token"] = str(result["user_id"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sign in failed: {exc}")
        with tabs[1]:
            with st.form("signup_form", clear_on_submit=True):
                name = st.text_input("Full name", placeholder="Your full name")
                email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="signup_password", placeholder="Min. 8 characters")
                signup_clicked = st.form_submit_button("Create Account", use_container_width=True)
            if signup_clicked:
                try:
                    result = register_user(name, email, password, config)
                    st.session_state.authenticated = True
                    st.session_state.active_user_id = result["user_id"]
                    st.session_state.active_user_name = result["full_name"]
                    st.session_state.active_user_email = result["email"]
                    st.query_params["auth_token"] = str(result["user_id"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Signup failed: {exc}")

    # Feature highlights below auth
    st.markdown(
        """<div style="display: flex; justify-content: center; gap: 40px; margin-top: 32px; flex-wrap: wrap;">
        <div style="text-align: center; max-width: 160px;">
            <div style="font-size: 1.6rem;">📄</div>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Upload documents &amp; get smart summaries</p>
        </div>
        <div style="text-align: center; max-width: 160px;">
            <div style="font-size: 1.6rem;">📝</div>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">AI-generated quizzes with instant feedback</p>
        </div>
        <div style="text-align: center; max-width: 160px;">
            <div style="font-size: 1.6rem;">📊</div>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Track your progress with detailed analytics</p>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _process_single_document(file_name: str, file_bytes: bytes, config, user_id) -> dict:
    """Process one document from its raw bytes. Thread-safe — no UploadedFile needed."""
    from modules import summarizer, utils

    filename_lower = file_name.lower()
    file_topic = Path(file_name).stem.replace("_", " ").replace("-", " ").strip() or "general_document"

    # Save raw file to disk
    raw_dir = Path(config.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / timestamped_name(file_name)
    raw_path.write_bytes(file_bytes)
    raw_saved_path = str(raw_path)

    # Extract text
    text = (
        utils.extract_text_from_pdf_bytes(file_bytes)
        if filename_lower.endswith(".pdf")
        else file_bytes.decode("utf-8")
    )
    chunks = utils.chunk_text(text)
    language = summarizer.detect_language(text)
    document = get_or_create_document(
        user_id, file_name,
        Path(file_name).suffix.lower(),
        file_topic, text, language, config,
    )
    return {
        "file_topic": file_topic,
        "filename": file_name,
        "text": text,
        "chunks": chunks,
        "language": language,
        "document": document,
        "raw_saved_path": raw_saved_path,
    }


def handle_upload(config) -> None:
    from modules import utils, vectorstore
    import concurrent.futures

    uploaded_files = st.sidebar.file_uploader(
        "📄 Upload Documents",
        type=["pdf", "txt", "csv", "json", "xlsx"],
        accept_multiple_files=True,
        help="Upload up to 6 files: PDF, TXT, CSV, JSON, or XLSX",
    )
    if not uploaded_files:
        return

    user_id = current_user_id()

    doc_files = [f for f in uploaded_files if f.name.lower().endswith((".pdf", ".txt"))]
    if not doc_files:
        st.sidebar.warning("Please upload PDF or TXT files.")
        return

    # Prevent destructive re-processing on every Streamlit rerun
    current_upload_signature = hash(tuple((f.name, f.size) for f in doc_files))
    if st.session_state.get("last_upload_signature") == current_upload_signature:
        return  # We already processed this exact set of files!
    
    st.session_state.last_upload_signature = current_upload_signature

    # ── Process documents in parallel ────────────────────────────────────────
    # Read all file bytes upfront in the main thread for thread safety
    file_payloads = []
    for f in doc_files:
        try:
            file_payloads.append((f.name, f.getvalue()))
        except Exception as exc:
            st.sidebar.error(f"❌ Could not read '{f.name}': {exc}")

    if not file_payloads:
        return

    results: list[dict] = []
    errors: list[tuple[str, str]] = []
    progress = st.sidebar.progress(0, text="Processing documents…")

    max_workers = min(4, len(file_payloads))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_process_single_document, name, fbytes, config, user_id): name
            for name, fbytes in file_payloads
        }
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            done_count += 1
            progress.progress(
                done_count / len(file_payloads),
                text=f"Processed {done_count}/{len(file_payloads)} files…",
            )
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append((future_map[future], str(exc)))

    progress.empty()

    if results:
        # Merge all chunks into a single vectorstore for cross-doc search
        all_chunks: list[str] = []
        for r in results:
            all_chunks.extend(r["chunks"])
        if all_chunks:
            vectorstore.build_vectorstore(all_chunks)

        # Build combined text with clear document separators
        doc_sections: list[str] = []
        for r in results:
            header = f"===== DOCUMENT: {r['filename']} ====="
            doc_sections.append(f"{header}\n\n{r['text']}")
        combined_text = "\n\n\n".join(doc_sections)

        with open(DOC_PATH, "w", encoding="utf-8") as fh:
            fh.write(combined_text)

        # Collect all individual doc IDs and names
        all_doc_ids = [int(r["document"]["id"]) for r in results]
        all_doc_names = [r["filename"] for r in results]
        all_topics = [r["file_topic"] for r in results]

        if len(results) == 1:
            single = results[0]
            active_doc_id = int(single["document"]["id"])
            active_doc_name = single["filename"]
            active_doc_topic = single["file_topic"]
        else:
            combined_filename = " + ".join(all_doc_names)
            combined_topic = ", ".join(dict.fromkeys(all_topics))
            combined_language = results[0]["language"]
            combined_doc = get_or_create_document(
                user_id,
                combined_filename,
                ".multi",
                combined_topic,
                combined_text,
                combined_language,
                config,
            )
            active_doc_id = int(combined_doc["id"])
            active_doc_name = combined_filename
            active_doc_topic = combined_topic

        st.session_state.current_document_topic = active_doc_topic
        st.session_state.current_document_name = active_doc_name
        st.session_state.current_document_id = active_doc_id
        st.session_state.uploaded_doc_ids = all_doc_ids
        st.session_state.uploaded_doc_names = all_doc_names
        st.session_state.summary_result = None
        st.session_state.quiz_package = None
        st.session_state.chat_session_id = None

        for r in results:
            activity_metadata = {
                "filename": r["filename"],
                "language": r["language"],
                "topics": [r["file_topic"]],
                "raw_path": r["raw_saved_path"],
            }
            log_user_activity(user_id, r["file_topic"], "document_uploaded", activity_metadata, config=config)
            log_event(
                user_id, "document_uploaded", activity_metadata, config=config,
                activity_type="document_upload",
                resource_id=str(r["document"]["id"]),
                metadata=activity_metadata,
                topics=[r["file_topic"]],
                session_id=str(r["document"]["id"]),
            )

        noun = "document" if len(results) == 1 else "documents"
        st.sidebar.success(f"✅ {len(results)} {noun} uploaded and indexed.")

    for fname, err in errors:
        st.sidebar.error(f"❌ {fname}: {err}")


def ensure_chat_session(config) -> int | None:
    if st.session_state.chat_session_id:
        return st.session_state.chat_session_id
    user_id = current_user_id()
    if not user_id:
        return None
    session_id = create_chat_session(
        user_id,
        title=st.session_state.current_document_name or "General Chat",
        topic=current_document_topic(),
        document_id=current_document_id(),
        config=config,
    )
    st.session_state.chat_session_id = session_id
    return session_id


def render_chatbot_sidebar(config) -> None:
    from modules import chatbot_rag

    st.sidebar.markdown("## Chatbot")
    user_id = current_user_id()
    if not user_id:
        return

    session_id = ensure_chat_session(config)
    sessions = list_chat_sessions(user_id, config)
    current_doc_id = current_document_id()
    if current_doc_id:
        filtered_sessions = [session for session in sessions if session.get("document_id") == current_doc_id]
        sessions = filtered_sessions or sessions

    if st.sidebar.button("New Chat Session"):
        st.session_state.chat_session_id = None
        session_id = ensure_chat_session(config)
        sessions = list_chat_sessions(user_id, config)
        if current_doc_id:
            filtered_sessions = [session for session in sessions if session.get("document_id") == current_doc_id]
            sessions = filtered_sessions or sessions
        st.rerun()

    if sessions:
        labels = [f"{session['title']} ({session['updated_at'][:16]})" for session in sessions]
        ids = [int(session["id"]) for session in sessions]
        default_index = ids.index(session_id) if session_id in ids else 0
        selected_label = st.sidebar.selectbox("Conversation", labels, index=default_index)
        st.session_state.chat_session_id = ids[labels.index(selected_label)]
        session_id = st.session_state.chat_session_id

    history_rows = list_chat_messages(session_id, config, limit=40) if session_id else []
    history = [{"role": row["role"], "content": row["message_text"]} for row in history_rows]
    doc_text = load_document_text() or ""

    chat_mode_label = st.sidebar.selectbox("Answer style", list(CHAT_ANSWER_MODES.keys()), index=0)
    question = st.sidebar.text_area("Ask about the uploaded document", key="sidebar_chat_input")
    if st.sidebar.button("Send Chat Question"):
        response = chatbot_rag.chatbot_respond(
            question,
            history=history,
            answer_mode=CHAT_ANSWER_MODES[chat_mode_label],
            document_text=doc_text,
        )
        if question.strip() and session_id is not None:
            answer_text = response["answer"]
            add_chat_message(session_id, user_id, "user", question, config=config)
            assistant_message_id = add_chat_message(
                session_id,
                user_id,
                "assistant",
                answer_text,
                config=config,
                confidence_score=response["confidence"],
                retrieval_metadata={"sources": response["sources"]},
            )
            st.session_state.last_assistant_message_id = assistant_message_id
            log_chat_event(user_id, current_document_topic(), question, answer_text, config=config)
            log_event(
                user_id,
                "chat_message",
                {"question": question, "confidence": response["confidence"]},
                config=config,
                activity_type="chat_interaction",
                resource_id=str(session_id),
                metadata={"topics": [current_document_topic()], "confidence": response["confidence"]},
                topics=[current_document_topic()],
                session_id=str(session_id),
            )
            st.rerun()

    if history_rows:
        st.sidebar.markdown("### Conversation")
        recent_rows = list(history_rows[-12:])
        for row in recent_rows:
            label = "You" if row["role"] == "user" else f"Bot ({row.get('confidence_score') or 0:.2f})"
            st.sidebar.markdown(f"**{label}:** {row['message_text']}")

    if st.session_state.last_assistant_message_id:
        rating = st.sidebar.radio("Rate latest bot answer", [1, 2, 3, 4, 5], horizontal=True, key="chat_rating")
        if st.sidebar.button("Save Chat Rating"):
            rate_chat_message(st.session_state.last_assistant_message_id, rating, config=config)
            log_event(
                user_id,
                "chat_feedback",
                {"rating": rating},
                config=config,
                activity_type="chat_feedback",
                resource_id=str(st.session_state.last_assistant_message_id),
                metadata={"rating": rating},
                topics=[current_document_topic()],
                session_id=str(session_id),
            )
            st.sidebar.success("Feedback saved.")


def render_sidebar_shell(config) -> None:
    user_name = st.session_state.active_user_name or "Student"
    st.sidebar.markdown(f"## {user_name}")
    st.sidebar.caption(st.session_state.active_user_email)
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    handle_upload(config)
    pages = ["Dashboard", "Summarizer", "Quiz", "Analytics"]
    current = st.session_state.current_page
    default_idx = pages.index(current) if current in pages else 0
    next_page = st.sidebar.radio("Navigate", pages, index=default_idx)
    if next_page != st.session_state.current_page:
        st.session_state.current_page = next_page
        log_event(current_user_id(), "page_view", {"page": next_page}, config=config, activity_type="navigation", resource_id=next_page)
    render_chatbot_sidebar(config)


def render_summarizer_page(config) -> None:
    from modules import summarizer

    st.header("Summarizer")
    doc_content = load_document_text()
    if not doc_content or not current_document_id():
        st.info("Upload PDF or TXT documents (up to 6) to use the summarizer.")
        return

    # Show active document names — one or multiple
    doc_names = st.session_state.get("uploaded_doc_names") or []
    if len(doc_names) > 1:
        st.caption(f"📄 Active documents ({len(doc_names)}): {', '.join(doc_names)}")
    else:
        st.caption(f"📄 Active document: {st.session_state.current_document_name}")
    mode = st.selectbox("Summary style", SUMMARY_MODES, format_func=lambda value: value.replace("_", " ").title())
    if st.button("Summarize Document"):
        with st.spinner("Building summary..."):
            requested_language = summarizer.detect_language(doc_content)
            st.session_state.summary_result = summarizer.summarize_document(
                current_user_id(),
                current_document_id(),
                doc_content,
                mode=mode,
                method="auto",
                target_language=requested_language,
                config=config,
            )
            study_minutes = estimate_study_minutes(doc_content)
            engagement_score = round(min(1.0, study_minutes / max(len(doc_content.split()) / 200, 1)), 2)
            log_user_activity(current_user_id(), current_document_topic(), "summary_requested", {"mode": mode, "method": "auto"}, config=config)
            log_study_session(
                current_user_id(),
                current_subject(),
                current_document_topic(),
                study_minutes,
                config,
                document_id=current_document_id(),
                engagement_score=engagement_score,
                completion_percentage=1.0,
            )
            log_event(
                current_user_id(),
                "summary_requested",
                {"method": "auto", "mode": mode},
                config=config,
                activity_type="summary_read",
                resource_id=str(current_document_id()),
                metadata={"method": "auto", "mode": mode},
                duration_seconds=study_minutes * 60,
                engagement_score=engagement_score,
                topics=[current_document_topic()],
                session_id=str(current_document_id()),
            )

    result = st.session_state.summary_result
    if result:
        if result.get("cached"):
            st.caption("Loaded from summary cache.")
        st.markdown("### Summary")
        st.markdown(strip_page_markers(result["summary_text"]))


def render_quiz_page(config) -> None:
    from modules import quiz_generator

    st.header("Quiz")
    doc_content = load_document_text()
    if not doc_content or not current_document_id():
        st.info("Upload PDF or TXT documents (up to 6) to generate a quiz.")
        return

    # Show active document names
    doc_names = st.session_state.get("uploaded_doc_names") or []
    if len(doc_names) > 1:
        st.caption(f"📄 Quiz covers {len(doc_names)} documents: {', '.join(doc_names)}")
    performance = get_user_performance_summary(current_user_id(), config)
    st.caption(f"Adaptive difficulty: {performance['recommended_difficulty']} based on average score {performance['avg_score']:.1f}%")
    controls = st.columns(2)
    num_questions = controls[0].slider("Number of questions", 2, 10, 5)
    selected_difficulty = controls[1].selectbox("Difficulty", ["adaptive", "easy", "medium", "hard"], index=0)
    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz..."):
            st.session_state.quiz_package = quiz_generator.generate_quiz_package(
                doc_content,
                num_questions,
                user_id=current_user_id(),
                topic=current_document_topic(),
                document_id=current_document_id(),
                difficulty_override=None if selected_difficulty == "adaptive" else selected_difficulty,
                config=config,
            )
            # Reset any previous answers when generating a new quiz
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.last_quiz_score = None
            # IMPORTANT: Clear legacy widget states to cleanly render the new quiz radio buttons
            for k in list(st.session_state.keys()):
                if k.startswith("q_"):
                    del st.session_state[k]
            log_event(current_user_id(), "quiz_generated", {"topic": current_document_topic(), "questions": num_questions}, config=config, activity_type="quiz_generation", resource_id=str(current_document_id()), metadata={"difficulty": st.session_state.quiz_package['difficulty']}, topics=[current_document_topic()])
            st.success("Quiz ready — pick an option for each question!")

    package = st.session_state.get("quiz_package")
    if not package:
        return

    questions = package["questions"]
    if package.get("topics"):
        st.caption("Quiz coverage: " + ", ".join(strip_page_markers(str(topic)) for topic in package["topics"][:6]))

    # Initialise tracking dicts in session state
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    # ── Progress indicator ───────────────────────────────────────────────
    valid_count = 0
    for q in questions:
        # Match the exact skip-logic used below
        q_options = [strip_page_markers(str(opt)) for opt in q.get("options", [])]
        if q_options and len(q_options) >= 2:
            valid_count += 1
            
    total_questions = valid_count
    answered_count = len(st.session_state.quiz_answers)
    st.progress(answered_count / total_questions if total_questions > 0 else 0, text=f"Answered {answered_count} of {total_questions}")

    # Callback to handle answer securely before rerender
    def handle_answer_change(idx_val, ans_key, corr_ans):
        user_choice = st.session_state.get(f"q_{idx_val}")
        if user_choice is not None:
            st.session_state.quiz_answers[ans_key] = {
                "user_answer": user_choice,
                "correct": user_choice.strip().lower() == corr_ans.lower(),
            }

    # ── Render each MCQ question with instant feedback ───────────────────
    for idx, question_data in enumerate(questions):
        answer_key = f"answered_{idx}"
        already_answered = answer_key in st.session_state.quiz_answers
        correct_answer = strip_page_markers(str(question_data.get("answer", ""))).strip()
        options = [strip_page_markers(str(opt)) for opt in question_data.get("options", [])]

        if not options or len(options) < 2:
            st.warning(f"Question {idx + 1} has no valid options — skipped.")
            continue

        # Difficulty badge
        diff = question_data.get("difficulty", "medium")
        diff_colors = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
        diff_badge = diff_colors.get(diff, "🟡")

        st.markdown(f"**Q{idx + 1}** {diff_badge} {strip_page_markers(str(question_data['question']))}")

        # A single st.radio maintains state across reruns intrinsically via 'key'
        st.radio(
            "Select your answer:",
            options,
            index=None,
            key=f"q_{idx}",
            label_visibility="collapsed",
            disabled=already_answered,
            on_change=handle_answer_change,
            args=(idx, answer_key, correct_answer)
        )

        # Display persistent feedback
        if already_answered:
            result = st.session_state.quiz_answers[answer_key]
            if result["correct"]:
                st.markdown(
                    f"""<div style="background: #dcfce7; border-left: 4px solid #22c55e; padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
                    <span style="color: #166534; font-weight: 600;">✅ Correct!</span></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
                    <span style="color: #991b1b; font-weight: 600;">❌ Incorrect.</span>
                    <span style="color: #991b1b;"> The correct answer is: <strong>{correct_answer}</strong></span></div>""",
                    unsafe_allow_html=True,
                )
            explanation = question_data.get("explanation")
            if explanation:
                st.caption(f"💡 {strip_page_markers(str(explanation))}")

        st.markdown("---")

    # ── Final score section ──────────────────────────────────────────────
    answered_count = len(st.session_state.quiz_answers)

    if answered_count > 0:
        correct_count = sum(1 for v in st.session_state.quiz_answers.values() if v["correct"])
        score_percent = round((correct_count / total_questions) * 100, 2)
        # Color the score based on performance
        if score_percent >= 80:
            score_color = "#22c55e"
        elif score_percent >= 50:
            score_color = "#eab308"
        else:
            score_color = "#ef4444"
        st.markdown(
            f"""<div style="text-align: center; padding: 16px; background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,250,252,0.9)); border-radius: 16px; border: 1px solid rgba(0,0,0,0.08); margin: 12px 0;">
            <span style="font-size: 1.8rem; font-weight: 700; color: {score_color};">{correct_count}/{total_questions}</span>
            <span style="font-size: 1.1rem; color: #64748b; margin-left: 8px;">({score_percent}%)</span></div>""",
            unsafe_allow_html=True,
        )

    if answered_count == total_questions and not st.session_state.quiz_submitted:
        # Auto-save results once all questions are answered
        st.session_state.quiz_submitted = True
        correct_count = sum(1 for v in st.session_state.quiz_answers.values() if v["correct"])
        score_percent = round((correct_count / total_questions) * 100, 2)
        st.session_state.last_quiz_score = score_percent

        struggled_ids: list[int] = []
        for i, q in enumerate(questions):
            key = f"answered_{i}"
            if key in st.session_state.quiz_answers and not st.session_state.quiz_answers[key]["correct"]:
                if q.get("question_id"):
                    struggled_ids.append(int(q["question_id"]))

        save_quiz_result(
            current_user_id(),
            current_subject(),
            current_document_topic(),
            correct_count,
            total_questions,
            config,
            document_id=current_document_id(),
            difficulty_level=package["difficulty"],
            question_types=["multiple_choice"] * total_questions,
            question_set_json=questions,
        )
        for q in questions:
            if q.get("question_id"):
                update_question_quality(int(q["question_id"]), int(q["question_id"]) in struggled_ids, config)
        log_quiz_attempt(current_user_id(), current_document_topic(), score_percent, total_questions, quiz_id=f"quiz-{uuid.uuid4().hex[:8]}", config=config)
        log_user_activity(current_user_id(), current_document_topic(), "quiz_submitted", {"questions": total_questions, "difficulty": package['difficulty']}, score_percent, config)
        log_event(current_user_id(), "quiz_submitted", {"score_percent": score_percent}, config=config, activity_type="quiz_attempt", resource_id=str(current_document_id()), metadata={"difficulty": package['difficulty'], "question_types": ["multiple_choice"]}, topics=[current_document_topic()], skill_level=package['difficulty'], completion_percentage=1.0)
        st.balloons()


def render_dashboard_page(config) -> None:
    """Student home screen — at-a-glance overview, quick stats, and recent activity."""
    from datetime import datetime as dt, timedelta

    user_id = current_user_id()
    user_name = st.session_state.active_user_name or "Student"

    # ── Fetch data ───────────────────────────────────────────────────────
    quiz_df = _cached_quiz_df(config)
    study_df = _cached_study_df(config)
    documents_df = _cached_documents_df(config)
    events_df = _cached_events_df(config, limit=200)

    user_quiz = quiz_df[quiz_df["user_id"] == user_id].copy() if not quiz_df.empty else pd.DataFrame()
    user_study = study_df[study_df["user_id"] == user_id].copy() if not study_df.empty else pd.DataFrame()
    user_docs = documents_df[documents_df["user_id"] == user_id].copy() if not documents_df.empty else pd.DataFrame()
    user_events = events_df[events_df["user_id"] == user_id].copy() if not events_df.empty else pd.DataFrame()

    # ── Compute stats ────────────────────────────────────────────────────
    total_quizzes = len(user_quiz)
    avg_score = round(float(user_quiz["score_percent"].mean()), 1) if total_quizzes > 0 else 0.0
    total_study_mins = int(user_study["time_spent"].sum()) if not user_study.empty else 0
    total_docs = len(user_docs)

    # Performance level
    if total_quizzes == 0:
        level_label, level_icon = "Beginner", "🌱"
    elif avg_score >= 85:
        level_label, level_icon = "Expert", "🏆"
    elif avg_score >= 70:
        level_label, level_icon = "Advanced", "⭐"
    elif avg_score >= 50:
        level_label, level_icon = "Intermediate", "📘"
    else:
        level_label, level_icon = "Beginner", "🌱"

    # Study streak (consecutive days with activity)
    streak = 0
    if not user_events.empty and "created_at" in user_events.columns:
        event_dates_raw = pd.to_datetime(user_events["created_at"], errors="coerce")
        event_dates = sorted(set(event_dates_raw.dropna().dt.date), reverse=True)
        today = dt.now().date()
        if event_dates and (event_dates[0] == today or event_dates[0] == today - timedelta(days=1)):
            streak = 1
            for i in range(1, len(event_dates)):
                if event_dates[i] == event_dates[i - 1] - timedelta(days=1):
                    streak += 1
                else:
                    break

    # ── Welcome Banner ───────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%); padding: 28px 32px; border-radius: 20px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);">
        <h1 style="color: #ffffff !important; margin: 0 0 6px 0; font-size: 1.8rem; font-weight: 800;">Welcome back, {user_name}! 👋</h1>
        <p style="color: rgba(255,255,255,0.9) !important; margin: 0; font-size: 1rem; font-weight: 500;">{level_icon} {level_label} • {dt.now().strftime('%A, %B %d, %Y')}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Quick Stats ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Quizzes Taken", total_quizzes)
    c2.metric("🎯 Avg Score", f"{avg_score}%")
    c3.metric("⏱️ Study Time", f"{total_study_mins} min")
    c4.metric("📄 Documents", total_docs)

    # ── Two-column: Recent Quiz + Study Streak ───────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📊 Last Quiz")
        if not user_quiz.empty:
            user_quiz["created_at"] = pd.to_datetime(user_quiz["created_at"], errors="coerce")
            latest = user_quiz.sort_values("created_at", ascending=False).iloc[0]
            score = latest["score_percent"]
            topic = latest.get("topic", "N/A")
            difficulty = latest.get("difficulty_level", "N/A")
            score_color = "#22c55e" if score >= 80 else ("#eab308" if score >= 50 else "#ef4444")

            st.markdown(
                f"""<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 20px; text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 700; color: {score_color};">{score:.0f}%</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">{topic}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Difficulty: {difficulty}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Trend vs previous
            if len(user_quiz) >= 2:
                prev_score = user_quiz.sort_values("created_at", ascending=False).iloc[1]["score_percent"]
                delta = score - prev_score
                if delta > 0:
                    st.markdown(f"<p style='color: #22c55e; text-align: center; margin-top: 8px;'>↑ +{delta:.0f}% from previous quiz</p>", unsafe_allow_html=True)
                elif delta < 0:
                    st.markdown(f"<p style='color: #ef4444; text-align: center; margin-top: 8px;'>↓ {delta:.0f}% from previous quiz</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #64748b; text-align: center; margin-top: 8px;'>→ Same as previous quiz</p>", unsafe_allow_html=True)
        else:
            st.info("Take your first quiz to see results here!")

    with col_right:
        st.markdown("### 🔥 Study Streak")
        if streak > 0:
            flame_bar = "🔥" * min(streak, 10)
            st.markdown(
                f"""<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 20px; text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 700; color: #f97316;">{streak} day{'s' if streak != 1 else ''}</div>
                <div style="font-size: 1.2rem; margin-top: 4px;">{flame_bar}</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Keep it going!</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 20px; text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 700; color: #94a3b8;">0 days</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Start studying today to build a streak!</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Quick Actions ────────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Actions")
    act1, act2, act3 = st.columns(3)
    if act1.button("📝 Take a Quiz", use_container_width=True):
        st.session_state.current_page = "Quiz"
        st.rerun()
    if act2.button("📄 Upload Document", use_container_width=True):
        st.session_state.current_page = "Summarizer"
        st.rerun()
    if act3.button("📊 View Analytics", use_container_width=True):
        st.session_state.current_page = "Analytics"
        st.rerun()

    # ── Recent Activity Feed ─────────────────────────────────────────────
    st.markdown("### 🕒 Recent Activity")
    if not user_events.empty:
        user_events["created_at"] = pd.to_datetime(user_events["created_at"], errors="coerce")
        recent = user_events.sort_values("created_at", ascending=False).head(8)

        activity_icons = {
            "quiz_submitted": "📝", "quiz_attempt": "📝", "quiz_generated": "🎲",
            "summary_requested": "📋", "summary_read": "📋",
            "document_uploaded": "📄", "document_upload": "📄",
            "chat_message": "💬", "chat_interaction": "💬",
            "navigation": "🔀", "page_view": "🔀",
        }

        for _, row in recent.iterrows():
            event_type = row.get("event_type", row.get("activity_type", "activity"))
            icon = activity_icons.get(event_type, "📌")
            timestamp = row["created_at"]
            time_str = timestamp.strftime("%b %d, %I:%M %p") if pd.notna(timestamp) else ""

            # Build description
            event_label = str(event_type).replace("_", " ").title()
            metadata = {}
            try:
                metadata = json.loads(row.get("event_data", "{}") or "{}")
            except Exception:
                pass
            detail = ""
            if "score_percent" in metadata:
                detail = f" — Score: {metadata['score_percent']}%"
            elif "mode" in metadata:
                detail = f" — {metadata['mode']}"

            st.markdown(
                f"""<div style="display: flex; align-items: center; padding: 8px 12px; border-bottom: 1px solid rgba(0,0,0,0.05);">
                <span style="font-size: 1.2rem; margin-right: 10px;">{icon}</span>
                <div style="flex: 1;">
                    <span style="color: #0f172a; font-weight: 500;">{event_label}</span>
                    <span style="color: #64748b; font-size: 0.85rem;">{detail}</span>
                </div>
                <span style="color: #94a3b8; font-size: 0.75rem;">{time_str}</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No activity yet — upload a document or take a quiz to get started!")


def render_analytics_page(config) -> None:
    """Deep-dive analytics page — quiz trends, knowledge map, and actionable recommendations."""
    st.header("📊 Learning Analytics")
    user_id = current_user_id()

    # ── Fetch data ───────────────────────────────────────────────────────
    quiz_df = _cached_quiz_df(config)
    study_df = _cached_study_df(config)
    documents_df = _cached_documents_df(config)
    summaries_df = _cached_summary_df(config)

    user_quiz = quiz_df[quiz_df["user_id"] == user_id].copy() if not quiz_df.empty else pd.DataFrame()
    user_study = study_df[study_df["user_id"] == user_id].copy() if not study_df.empty else pd.DataFrame()
    user_docs = documents_df[documents_df["user_id"] == user_id].copy() if not documents_df.empty else pd.DataFrame()
    user_summaries = summaries_df[summaries_df["user_id"] == user_id].copy() if not summaries_df.empty else pd.DataFrame()

    total_quizzes = len(user_quiz)
    total_docs = len(user_docs)
    avg_score = round(float(user_quiz["score_percent"].mean()), 1) if total_quizzes > 0 else 0.0

    if total_quizzes == 0 and total_docs == 0:
        st.info("No activity yet! Upload a document, generate a summary, or take a quiz to start seeing your analytics.")
        return

    # ── 1. Quiz Performance Over Time ────────────────────────────────────
    if total_quizzes > 0:
        st.markdown("### 📈 Quiz Performance Over Time")
        user_quiz["created_at"] = pd.to_datetime(user_quiz["created_at"], errors="coerce")
        quiz_timeline = user_quiz.dropna(subset=["created_at"]).sort_values("created_at")

        if len(quiz_timeline) >= 2:
            chart_data = quiz_timeline[["created_at", "score_percent"]].set_index("created_at")
            st.line_chart(chart_data, use_container_width=True)

            # Trend analysis
            first_half = quiz_timeline["score_percent"].iloc[: len(quiz_timeline) // 2].mean()
            second_half = quiz_timeline["score_percent"].iloc[len(quiz_timeline) // 2 :].mean()
            delta = round(second_half - first_half, 1)
            if delta > 2:
                st.markdown(
                    f"""<div style="background: #dcfce7; border-left: 4px solid #22c55e; padding: 12px 16px; border-radius: 8px;">
                    <span style="color: #166534; font-weight: 600;">🚀 You're improving!</span>
                    <span style="color: #166534;"> Your recent scores are <strong>{delta}%</strong> higher than your earlier attempts.</span></div>""",
                    unsafe_allow_html=True,
                )
            elif delta < -5:
                st.markdown(
                    f"""<div style="background: #fef3c7; border-left: 4px solid #eab308; padding: 12px 16px; border-radius: 8px;">
                    <span style="color: #92400e; font-weight: 600;">⚠️ Scores dipping.</span>
                    <span style="color: #92400e;"> Recent scores dropped by <strong>{abs(delta)}%</strong>. Consider reviewing the material again.</span></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 8px;">
                    <span style="color: #1e40af; font-weight: 600;">📊 Steady performance.</span>
                    <span style="color: #1e40af;"> Your scores are consistent — keep practising to push higher!</span></div>""",
                    unsafe_allow_html=True,
                )

            # Score summary metrics
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Average", f"{avg_score}%")
            sc2.metric("Best", f"{quiz_timeline['score_percent'].max():.0f}%")
            sc3.metric("Latest", f"{quiz_timeline.iloc[-1]['score_percent']:.0f}%")
            sc4.metric("Total Quizzes", total_quizzes)
        else:
            row = quiz_timeline.iloc[0]
            st.metric("Your Quiz Score", f"{row['score_percent']}%", help=f"Topic: {row.get('topic', 'N/A')}")
            st.caption("Take more quizzes to see performance trends!")

    # ── 2. Knowledge Map ─────────────────────────────────────────────────
    if total_quizzes > 0 and "topic" in user_quiz.columns:
        st.markdown("### 🎯 Knowledge Map")
        st.caption("Your mastery level across all topics — weakest topics appear first so you know what to focus on.")

        topic_stats = (
            user_quiz.groupby("topic", as_index=False)
            .agg(
                avg_score=("score_percent", "mean"),
                best_score=("score_percent", "max"),
                latest_score=("score_percent", "last"),
                attempts=("topic", "size"),
            )
        )
        topic_stats["avg_score"] = topic_stats["avg_score"].round(1)
        topic_stats["best_score"] = topic_stats["best_score"].round(1)
        topic_stats = topic_stats.sort_values("avg_score", ascending=True)

        for _, row in topic_stats.iterrows():
            topic_name = row["topic"]
            avg = row["avg_score"]
            best = row["best_score"]
            attempts = int(row["attempts"])

            # Mastery level
            if avg >= 85:
                mastery, mastery_color, mastery_bg = "Mastered", "#22c55e", "#dcfce7"
            elif avg >= 70:
                mastery, mastery_color, mastery_bg = "Proficient", "#3b82f6", "#dbeafe"
            elif avg >= 45:
                mastery, mastery_color, mastery_bg = "Learning", "#eab308", "#fef3c7"
            else:
                mastery, mastery_color, mastery_bg = "Needs Work", "#ef4444", "#fee2e2"

            # Trend arrow (compare first half vs second half of attempts for this topic)
            topic_quizzes = user_quiz[user_quiz["topic"] == topic_name].sort_values("created_at")
            if len(topic_quizzes) >= 2:
                first = topic_quizzes["score_percent"].iloc[: len(topic_quizzes) // 2].mean()
                second = topic_quizzes["score_percent"].iloc[len(topic_quizzes) // 2 :].mean()
                if second - first > 3:
                    trend_arrow, trend_text = "↑", "improving"
                elif first - second > 3:
                    trend_arrow, trend_text = "↓", "declining"
                else:
                    trend_arrow, trend_text = "→", "steady"
            else:
                trend_arrow, trend_text = "—", "not enough data"

            st.markdown(
                f"""<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 16px 20px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 600; font-size: 1rem; color: #0f172a;">{topic_name}</span>
                        <span style="background: {mastery_bg}; color: {mastery_color}; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-left: 10px;">{mastery}</span>
                    </div>
                    <span style="color: #64748b; font-size: 0.85rem;">{attempts} attempt{'s' if attempts != 1 else ''} • Trend: {trend_arrow} {trend_text}</span>
                </div>
                <div style="background: #f1f5f9; border-radius: 8px; height: 10px; overflow: hidden;">
                    <div style="background: {mastery_color}; height: 100%; width: {min(avg, 100)}%; border-radius: 8px; transition: width 0.3s;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 6px; color: #64748b; font-size: 0.8rem;">
                    <span>Avg: {avg}%</span>
                    <span>Best: {best}%</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── 3. Difficulty Breakdown ──────────────────────────────────────────
    if total_quizzes > 0 and "difficulty_level" in user_quiz.columns:
        valid_diff = user_quiz.dropna(subset=["difficulty_level"])
        if not valid_diff.empty:
            st.markdown("### 🏋️ Score by Difficulty")
            diff_stats = (
                valid_diff.groupby("difficulty_level", as_index=False)
                .agg(avg_score=("score_percent", "mean"), count=("difficulty_level", "size"))
            )
            diff_stats["avg_score"] = diff_stats["avg_score"].round(1)

            # Ordered display
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            diff_stats["sort_key"] = diff_stats["difficulty_level"].map(diff_order).fillna(9)
            diff_stats = diff_stats.sort_values("sort_key")

            diff_icons = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
            for _, row in diff_stats.iterrows():
                label = row["difficulty_level"].capitalize()
                icon = diff_icons.get(row["difficulty_level"], "⚪")
                pct = row["avg_score"]
                count = int(row["count"])
                bar_color = "#22c55e" if pct >= 75 else ("#eab308" if pct >= 50 else "#ef4444")

                st.markdown(
                    f"""<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #0f172a;">{icon} {label}</span>
                        <span style="color: #64748b; font-size: 0.85rem;">{pct}% avg • {count} quiz{'zes' if count != 1 else ''}</span>
                    </div>
                    <div style="background: #f1f5f9; border-radius: 6px; height: 8px; overflow: hidden;">
                        <div style="background: {bar_color}; height: 100%; width: {min(pct, 100)}%; border-radius: 6px;"></div>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── 4. Smart Recommendations ─────────────────────────────────────────
    st.markdown("### ⚡ Smart Recommendations")
    recommendations: list[tuple[str, str, str]] = []  # (icon, text, color)

    if total_quizzes > 0 and "topic" in user_quiz.columns:
        topic_stats = (
            user_quiz.groupby("topic", as_index=False)
            .agg(avg_score=("score_percent", "mean"), attempts=("topic", "size"))
        )
        # Weak topics
        weak_topics = topic_stats[topic_stats["avg_score"] < 60].sort_values("avg_score")
        for _, row in weak_topics.head(3).iterrows():
            recommendations.append((
                "📖",
                f"Revise **{row['topic']}** (avg {row['avg_score']:.0f}%) — re-read the summary and retry the quiz.",
                "#fee2e2",
            ))
        # Strong topics — bump difficulty
        strong_topics = topic_stats[topic_stats["avg_score"] >= 85]
        for _, row in strong_topics.head(2).iterrows():
            recommendations.append((
                "🚀",
                f"You've mastered **{row['topic']}** at {row['avg_score']:.0f}% — try a harder difficulty!",
                "#dcfce7",
            ))
        # General performance
        if avg_score >= 80 and total_quizzes >= 3:
            recommendations.append(("🌟", "Outstanding performance! You're consistently scoring above 80%.", "#dbeafe"))
        elif avg_score >= 60:
            recommendations.append(("💪", "Solid progress — focus on your weakest topics to improve further.", "#fef3c7"))

    if total_docs > 0 and total_quizzes == 0:
        recommendations.append(("📝", "You've uploaded documents but haven't taken any quizzes yet — try one!", "#fef3c7"))
    if total_docs > 0 and len(user_summaries) == 0:
        recommendations.append(("📋", "Generate a summary of your documents to reinforce understanding.", "#dbeafe"))
    if not recommendations:
        recommendations.append(("🎯", "Upload a document and take your first quiz to get personalised recommendations.", "#f1f5f9"))

    for icon, text, bg in recommendations:
        st.markdown(
            f"""<div style="background: {bg}; border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center;">
            <span style="font-size: 1.3rem; margin-right: 12px;">{icon}</span>
            <span style="color: #0f172a;">{text}</span></div>""",
            unsafe_allow_html=True,
        )

    # ── 5. Study Activity ────────────────────────────────────────────────
    if not user_study.empty:
        st.markdown("### 📚 Study Activity")
        total_study_mins = int(user_study["time_spent"].sum())
        st.caption(f"Total study time: **{total_study_mins} minutes** across {len(user_study)} sessions")

        user_study["created_at"] = pd.to_datetime(user_study["created_at"], errors="coerce")
        study_by_topic = (
            user_study.groupby("topic", as_index=False)["time_spent"]
            .sum()
            .sort_values("time_spent", ascending=False)
        )
        st.bar_chart(study_by_topic.set_index("topic"), use_container_width=True)

    # ── 6. Documents & Resources ─────────────────────────────────────────
    if not user_docs.empty:
        st.markdown("### 📁 Your Documents")
        display_cols = [c for c in ["filename", "topic", "language", "usage_count", "updated_at"] if c in user_docs.columns]
        st.dataframe(
            user_docs[display_cols].rename(columns={
                "filename": "Document",
                "topic": "Topic",
                "language": "Language",
                "usage_count": "Times Used",
                "updated_at": "Last Used",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── 7. Export ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("💾 Export your data"):
        export_table_name = st.selectbox(
            "Select table",
            ["quiz_results", "study_sessions", "documents", "summaries"],
        )
        export_df = export_table(export_table_name, config)
        if "user_id" in export_df.columns:
            export_df = export_df[export_df["user_id"] == user_id]
        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                "Download CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"my_{export_table_name}.csv",
                mime="text/csv",
            )
        with col_json:
            st.download_button(
                "Download JSON",
                export_df.to_json(orient="records", indent=2).encode("utf-8"),
                file_name=f"my_{export_table_name}.json",
                mime="application/json",
            )


def render_pipeline_page(config) -> None:
    from modules import analytics
    from learnmate_ai.spark_manager import spark_runtime_status

    st.header("Pipeline Ops")
    st.json(spark_runtime_status(config))
    st.json(database_status(config))
    pipeline_cols = st.columns(3)
    if pipeline_cols[0].button("Initialize Database"):
        initialize_database_schema(config)
        st.success("Database initialized.")
    if pipeline_cols[1].button("Run Spark Batch Pipeline"):
        try:
            from batch_processing.big_data_pipeline import run_batch_pipeline

            report = run_batch_pipeline(config)
            st.session_state.pipeline_report = report
            st.session_state.pipeline_summary = analytics.summarize_pipeline_report(report)
            st.success("Spark batch pipeline completed.")
        except Exception as exc:
            st.error(f"Spark pipeline failed: {exc}")
    if pipeline_cols[2].button("Persist Report To Database"):
        if not st.session_state.pipeline_report:
            st.warning("Run the Spark pipeline first.")
        else:
            result = persist_pipeline_report(st.session_state.pipeline_report, config)
            st.success(f"Pipeline metadata stored with event id {result['run_id']}.")
    if st.session_state.pipeline_report:
        report = st.session_state.pipeline_report
        st.markdown("### Pipeline Summary")
        if st.session_state.pipeline_summary:
            st.markdown(st.session_state.pipeline_summary)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Processed Records", report.get("records_processed", 0))
        metric_cols[1].metric("Log Records", report.get("log_records_ingested", 0))
        metric_cols[2].metric("DB Records", report.get("database_records_ingested", 0))
        runtime_cols = st.columns(3)
        runtime_cols[0].metric("Processing Seconds", report.get("processing_seconds", 0))
        runtime_cols[1].metric("Records / Second", report.get("records_per_second", 0))
        runtime_cols[2].metric("Scale", report.get("scale_classification", "unknown"))
        if report.get("bronze_paths"):
            st.markdown("### Bronze Zone")
            st.json(report["bronze_paths"])
        if report.get("silver_paths"):
            st.markdown("### Silver Zone")
            st.json(report["silver_paths"])
        if report.get("gold_paths"):
            st.markdown("### Gold Zone")
            st.json(report["gold_paths"])
        if report.get("topic_metrics_preview"):
            st.markdown("### Gold Topic Metrics")
            st.dataframe(pd.DataFrame(report["topic_metrics_preview"]), width="stretch")
        if report.get("user_engagement_preview"):
            st.markdown("### Gold User Engagement")
            st.dataframe(pd.DataFrame(report["user_engagement_preview"]), width="stretch")
        if report.get("daily_activity_preview"):
            st.markdown("### Gold Daily Activity")
            st.dataframe(pd.DataFrame(report["daily_activity_preview"]), width="stretch")
        if report.get("student_clusters_preview"):
            st.markdown("### Student Clusters")
            st.dataframe(pd.DataFrame(report["student_clusters_preview"]), width="stretch")
        if report.get("performance_predictions_preview"):
            st.markdown("### Performance Predictions")
            st.dataframe(pd.DataFrame(report["performance_predictions_preview"]), width="stretch")
        if report.get("learning_recommendations_preview"):
            st.markdown("### Learning Recommendations")
            st.dataframe(pd.DataFrame(report["learning_recommendations_preview"]), width="stretch")


def main() -> None:
    from modules import utils

    st.set_page_config(page_title="LearnMate AI", layout="wide")
    inject_theme()
    utils.ensure_directory("data")
    config = ensure_data_directories(get_config())

    # Only run heavy init once per session
    if not st.session_state.get("_db_initialized"):
        initialize_database_schema(config)
        ensure_log_files(config)
        st.session_state["_db_initialized"] = True

    init_state()

    if not st.session_state.authenticated:
        render_auth_page(config)
        return

    render_sidebar_shell(config)
    page = st.session_state.current_page
    if page == "Dashboard":
        render_dashboard_page(config)
    elif page == "Summarizer":
        render_summarizer_page(config)
    elif page == "Quiz":
        render_quiz_page(config)
    elif page == "Analytics":
        render_analytics_page(config)


if __name__ == "__main__":
    main()
