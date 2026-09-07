"""Personal food journal: image recognition, daily logging and weekly reports."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from io import BytesIO
from html import escape
from typing import Any

import pandas as pd
import streamlit as st 
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from calendar_utils import saturday_week_bounds

from auth import (
    AccountValidationError,
    generate_reset_code,
    hash_password,
    hash_reset_code,
    validate_email,
    validate_password,
    validate_username,
    verify_password,
    verify_reset_code,
)
from database import (
    complete_email_verification,
    complete_password_reset,
    create_email_verification_token,
    create_meal,
    create_password_reset_token,
    create_user,
    daily_summary,
    daily_totals,
    delete_meal,
    get_active_email_verification,
    get_active_password_reset,
    get_meal_items,
    get_meals_between,
    get_user_credentials,
    get_meals_for_date,
    get_user_by_email,
    get_user_by_id,
    get_user_for_login,
    init_db,
    meal_type_totals,
    record_password_reset_failure,
    record_email_verification_failure,
    top_foods,
    update_account_profile,
    update_daily_goal,
    update_password_hash,
)
from mailer import (
    reset_code_debug_enabled,
    send_email_verification_code,
    send_password_reset_code,
    smtp_is_configured,
)
from services.classifier import Prediction
from services.hybrid_classifier import HybridFoodClassifier, HybridPrediction
from services.labels_fa import LABELS_FA, search_food_labels, to_persian
from services.measurements import (
    amount_label_fa,
    calories_for_display_unit,
    default_amount,
    is_drink_label,
    max_amount,
    measure_unit_for_label,
    per_100_label_fa,
    short_unit_fa,
)
from services.usda import NutritionLookupError, NutritionResult, lookup_calories

load_dotenv()

st.set_page_config(
    page_title="دفترچه هوشمند تغذیه",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# UI state is kept in Streamlit Session State so it survives page navigation/reruns.
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "dark"
if "mobile_menu_open" not in st.session_state:
    st.session_state["mobile_menu_open"] = False

st.markdown(
    """
    <style>
    :root {
        --blue-500: #2f80ff;
        --blue-400: #58a6ff;
        --blue-300: #7cc4ff;
        --cyan: #25d6ff;
        --ink-950: #030711;
        --ink-900: #060b16;
        --ink-850: #09111f;
        --ink-800: #0b1626;
        --ink-750: #0f1d31;
        --line: rgba(88,166,255,.20);
        --line-strong: rgba(88,166,255,.38);
        --text: #f4f8ff;
        --muted: #91a5be;
        --success: #3ddc97;
        --danger: #ff6b7a;
        --shadow: 0 18px 55px rgba(0,0,0,.34);
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: right;
        color: var(--text) !important;
        background: var(--ink-950) !important;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% -5%, rgba(47,128,255,.28), transparent 28rem),
            radial-gradient(circle at 8% 18%, rgba(37,214,255,.10), transparent 24rem),
            linear-gradient(145deg, #030711 0%, #06101e 48%, #030711 100%) !important;
        min-height: 100vh;
    }

    /* The native Streamlit sidebar is intentionally removed since v6; v8 keeps the hamburger navigation and adds an icon-only theme switch.
       Navigation lives inside the page so collapse/overlay DOM bugs cannot occur. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: .6rem !important;
    }
    [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }

    .block-container {
        max-width: 1240px;
        padding: 1.15rem 1.35rem 3rem !important;
    }

    h1, h2, h3, h4, p, label, .stCaption { color: var(--text); }
    h1, h2, h3 { letter-spacing: -.02em; }
    a { color: var(--blue-300) !important; }
    hr { border-color: rgba(88,166,255,.12) !important; }

    /* v9: the clickable theme switch lives inside this real Streamlit navbar container. */
    .st-key-top_navbar {
        position: relative !important;
        margin: .25rem 0 .8rem !important;
        padding: .9rem 1rem !important;
        border: 1px solid var(--line) !important;
        border-radius: 22px !important;
        background:
            linear-gradient(115deg, rgba(47,128,255,.20), rgba(9,17,31,.97) 38%, rgba(37,214,255,.075)) !important;
        box-shadow: 0 18px 55px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.035) !important;
        overflow: hidden !important;
    }
    .st-key-top_navbar::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        border-radius: 999px;
        top: -145px;
        left: -95px;
        background: rgba(47,128,255,.13);
        pointer-events: none;
    }
    .st-key-top_navbar [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        flex-wrap: nowrap !important;
        gap: .7rem !important;
        direction: ltr !important;
    }
    .st-key-top_navbar .brand-wrap,
    .st-key-top_navbar .user-chip { direction: rtl !important; }
    .st-key-top_navbar [data-testid="column"] { min-width: 0 !important; }
    .st-key-top_navbar .brand-wrap,
    .st-key-top_navbar .user-chip { margin: 0 !important; }
    .st-key-top_navbar [data-testid="column"]:has(.st-key-theme_toggle) {
        z-index: 3 !important;
        flex: 0 0 52px !important;
        width: 52px !important;
        min-width: 52px !important;
    }

    .app-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: .25rem 0 .8rem;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background:
            linear-gradient(115deg, rgba(47,128,255,.20), rgba(9,17,31,.96) 36%, rgba(37,214,255,.07));
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
    }
    .app-topbar::after {
        content: "";
        position: absolute;
        inset: auto -65px -90px auto;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        background: rgba(47,128,255,.12);
        filter: blur(4px);
        pointer-events: none;
    }
    .brand-wrap { display: flex; align-items: center; gap: .85rem; min-width: 0; }
    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        font-size: 1.45rem;
        background: linear-gradient(135deg, #2f80ff, #1264e7 60%, #00b8e6);
        box-shadow: 0 10px 30px rgba(47,128,255,.32);
        flex: 0 0 auto;
    }
    .brand-title { font-weight: 900; font-size: 1.04rem; color: #fff; }
    .brand-subtitle { color: var(--muted); font-size: .82rem; margin-top: .16rem; }
    .user-chip {
        z-index: 1;
        display: flex;
        align-items: center;
        gap: .65rem;
        padding: .55rem .72rem;
        border-radius: 14px;
        border: 1px solid rgba(88,166,255,.22);
        background: rgba(3,7,17,.58);
        white-space: nowrap;
    }
    .user-dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--success);
        box-shadow: 0 0 14px rgba(61,220,151,.65);
    }
    .user-name { font-weight: 800; color: #fff; }
    .user-goal { color: var(--muted); font-size: .78rem; }

    .nav-caption {
        color: #7f96b1;
        font-size: .78rem;
        margin: .25rem .15rem .35rem;
    }
    .st-key-main_navigation [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .55rem !important;
        background: rgba(6,11,22,.78);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: .5rem;
        box-shadow: 0 12px 38px rgba(0,0,0,.20);
    }
    .st-key-main_navigation [role="radiogroup"] > label {
        margin: 0 !important;
        width: 100% !important;
        min-height: 46px;
        border-radius: 12px;
        padding: .55rem .65rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid transparent;
        background: transparent;
        transition: .15s ease;
        cursor: pointer;
    }
    .st-key-main_navigation [role="radiogroup"] > label:hover {
        background: rgba(47,128,255,.11);
        border-color: rgba(88,166,255,.18);
    }
    .st-key-main_navigation [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #1667e8, #2f80ff 64%, #147ab8);
        border-color: rgba(124,196,255,.38);
        box-shadow: 0 10px 26px rgba(47,128,255,.26);
    }
    .st-key-main_navigation [role="radiogroup"] > label:has(input:checked) p {
        color: white !important;
        font-weight: 850 !important;
    }
    .st-key-main_navigation [data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    .st-key-main_navigation p {
        margin: 0 !important;
        text-align: center !important;
        font-size: .88rem !important;
        color: #c7d7e9 !important;
    }

    /* Fallback styling for Streamlit builds that do not expose st-key-* classes. */
    div[data-testid="stRadio"] [role="radiogroup"] {
        gap: .5rem !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label {
        border: 1px solid rgba(88,166,255,.16);
        border-radius: 11px;
        padding: .45rem .62rem !important;
        background: rgba(9,18,33,.60);
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
        background: rgba(47,128,255,.11);
        border-color: rgba(88,166,255,.28);
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        opacity: .72;
    }

    .page-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.55rem 0 1.05rem;
    }
    .page-title-row { display: flex; align-items: center; gap: .75rem; }
    .page-icon {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: rgba(47,128,255,.14);
        border: 1px solid rgba(88,166,255,.24);
        color: var(--blue-300);
        font-size: 1.25rem;
        box-shadow: inset 0 0 22px rgba(47,128,255,.06);
    }
    .page-title {
        margin: 0;
        font-size: clamp(1.45rem, 2.5vw, 2.15rem);
        font-weight: 950;
        color: #fff;
    }
    .page-subtitle { color: var(--muted); margin-top: .25rem; font-size: .9rem; }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: .75rem;
        margin: .85rem 0 1rem;
    }
    .metric-card {
        min-height: 126px;
        padding: 1rem 1.05rem;
        border-radius: 19px;
        border: 1px solid var(--line);
        background:
            linear-gradient(150deg, rgba(17,34,57,.96), rgba(7,14,26,.97));
        box-shadow: 0 16px 42px rgba(0,0,0,.24);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #2f80ff, #25d6ff);
        opacity: .9;
    }
    .metric-label { color: #8fb5e5; font-size: .84rem; font-weight: 750; }
    .metric-value {
        color: #fff;
        font-size: clamp(1.45rem, 2.6vw, 2.05rem);
        font-weight: 950;
        margin-top: .72rem;
        direction: ltr;
        text-align: right;
        line-height: 1.05;
    }
    .metric-hint { color: #738aa5; font-size: .73rem; margin-top: .45rem; }

    .progress-panel {
        padding: .95rem 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(7,14,26,.76);
        margin: .35rem 0 1.2rem;
    }
    .progress-meta {
        display: flex;
        justify-content: space-between;
        gap: .8rem;
        color: #a9bfd8;
        font-size: .84rem;
        margin-bottom: .6rem;
    }
    .progress-track {
        height: 9px;
        background: #111a29;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.025);
    }
    .progress-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #1768e9, #2f80ff 52%, #25d6ff);
        box-shadow: 0 0 18px rgba(47,128,255,.36);
    }

    .hero-card, .soft-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: .6rem 0 1rem;
        color: #d9e8f8;
        line-height: 1.95;
        box-shadow: 0 14px 36px rgba(0,0,0,.18);
    }
    .hero-card {
        background: linear-gradient(125deg, rgba(47,128,255,.18), rgba(8,18,34,.94) 52%, rgba(37,214,255,.07));
        border-right: 4px solid #2f80ff;
    }
    .soft-card { background: rgba(9,18,33,.86); }
    .small-muted { color: var(--muted); font-size: .9rem; }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background: rgba(9,18,33,.88) !important;
        padding: .8rem .9rem !important;
    }
    div[data-testid="stMetric"] label { color: #93bce9 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; direction:ltr; text-align:right; }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 12px !important;
        min-height: 2.8rem;
        border: 1px solid rgba(88,166,255,.25) !important;
        background: linear-gradient(180deg, #10213a, #0b1729) !important;
        color: #eaf4ff !important;
        font-weight: 800 !important;
        transition: .14s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(88,166,255,.55) !important;
        box-shadow: 0 10px 28px rgba(47,128,255,.18) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(100deg, #1768e9, #2f80ff 58%, #087eb8) !important;
        color: #fff !important;
        border-color: rgba(124,196,255,.32) !important;
        box-shadow: 0 12px 30px rgba(47,128,255,.24) !important;
    }
    .st-key-logout_top button {
        background: rgba(255,107,122,.07) !important;
        border-color: rgba(255,107,122,.22) !important;
        color: #ffc3ca !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        background: #081424 !important;
        border-color: rgba(88,166,255,.22) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
    }
    input, textarea { color: var(--text) !important; caret-color: var(--blue-400) !important; }
    input::placeholder, textarea::placeholder { color: #657c96 !important; opacity: 1 !important; }
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"] > div {
        background: #081424 !important;
        border-color: rgba(88,166,255,.30) !important;
    }
    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: #f4f8ff !important;
        -webkit-text-fill-color: #f4f8ff !important;
        opacity: 1 !important;
        font-weight: 650 !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #8097b2 !important;
        -webkit-text-fill-color: #8097b2 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }
    [data-baseweb="popover"] > div,
    [role="listbox"] {
        background: #081424 !important;
        border: 1px solid var(--line-strong) !important;
        color: #eef6ff !important;
    }
    [role="option"] { color: #eef6ff !important; }
    [role="option"]:hover { background: rgba(47,128,255,.16) !important; }

    [data-testid="stFileUploader"] {
        background: rgba(8,20,36,.78) !important;
        border: 1px dashed rgba(88,166,255,.40) !important;
        border-radius: 17px !important;
        padding: .35rem !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }
    [data-baseweb="tab-list"] { gap: .4rem; }
    [data-baseweb="tab"] {
        background: rgba(9,18,33,.60) !important;
        border-radius: 10px 10px 0 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid rgba(88,166,255,.16) !important;
    }

    .st-key-theme_toggle button {
        width: 46px !important;
        min-width: 46px !important;
        max-width: 46px !important;
        min-height: 46px !important;
        padding: 0 !important;
        border-radius: 14px !important;
        background: #0d1a2d !important;
        border-color: rgba(88,166,255,.36) !important;
        color: #ffd166 !important;
        font-size: 1.28rem !important;
        box-shadow: 0 8px 22px rgba(0,0,0,.20) !important;
    }
    .st-key-theme_toggle button:hover {
        background: #132640 !important;
        border-color: rgba(88,166,255,.68) !important;
        color: #ffe39a !important;
        box-shadow: 0 10px 24px rgba(47,128,255,.16) !important;
    }
    .st-key-mobile_menu_toggle button {
        background: rgba(47,128,255,.12) !important;
        border-color: rgba(88,166,255,.32) !important;
    }

    @media (min-width: 821px) {
        .st-key-mobile_menu_toggle { display: none !important; }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) {
            align-items: center !important;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(1) {
            display: none !important;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(2) {
            flex: 1 1 auto !important;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(3) {
            flex: 0 0 175px !important;
            width: 175px !important;
            min-width: 175px !important;
        }
    }

    @media (max-width: 820px) {
        .block-container { padding: .75rem .72rem 2rem !important; }
        .app-topbar { padding: .8rem; border-radius: 17px; align-items: flex-start; }
        .brand-logo { width: 42px; height: 42px; border-radius: 13px; }
        .brand-subtitle { display: none; }
        .user-chip { padding: .45rem .55rem; }
        .user-goal { display: none; }
        /* In v8 the mobile navigation is opened by a hamburger button.
           Runtime CSS decides whether this block is visible. */
        .st-key-main_navigation [role="radiogroup"] {
            grid-template-columns: 1fr !important;
            gap: .42rem !important;
            padding: .55rem !important;
            border-radius: 16px !important;
        }
        .st-key-main_navigation [role="radiogroup"] > label {
            width: 100% !important;
            min-width: 0 !important;
            min-height: 48px !important;
            justify-content: flex-start !important;
        }
        .st-key-main_navigation [role="radiogroup"] > label p {
            text-align: right !important;
            width: 100% !important;
        }

        /* Mobile action row: hamburger + logout. Theme stays inside the navbar. */
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) {
            flex-wrap: nowrap !important;
            gap: .45rem !important;
            align-items: stretch !important;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(1),
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(3) {
            flex: 1 1 0 !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-mobile_menu_toggle) > [data-testid="column"]:nth-child(2) {
            display: none !important;
        }
        .st-key-mobile_menu_toggle button,
        .st-key-logout_top button {
            min-height: 44px !important;
            padding-left: .55rem !important;
            padding-right: .55rem !important;
            font-size: .82rem !important;
            white-space: nowrap !important;
        }

        .st-key-top_navbar {
            padding: .72rem .75rem !important;
            border-radius: 18px !important;
        }
        [data-testid="stMain"] .st-key-top_navbar [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: .45rem !important;
            align-items: center !important;
            direction: ltr !important;
        }
        [data-testid="stMain"] .st-key-top_navbar [data-testid="column"]:nth-child(1) {
            flex: 1 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [data-testid="stMain"] .st-key-top_navbar [data-testid="column"]:nth-child(2) {
            display: none !important;
        }
        [data-testid="stMain"] .st-key-top_navbar [data-testid="column"]:nth-child(3) {
            display: block !important;
            flex: 0 0 48px !important;
            width: 48px !important;
            min-width: 48px !important;
        }
        .st-key-top_navbar .st-key-theme_toggle button {
            width: 44px !important;
            min-width: 44px !important;
            max-width: 44px !important;
            min-height: 44px !important;
            padding: 0 !important;
            font-size: 1.18rem !important;
        }
        .page-head { margin-top: 1.15rem; }
        .page-icon { width: 40px; height: 40px; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0,1fr)); gap: .6rem; }
        .metric-card { min-height: 110px; padding: .85rem; }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: .6rem !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        .stButton > button, .stFormSubmitButton > button { width: 100% !important; min-height: 46px; }
        [data-baseweb="tab-list"] { overflow-x: auto; white-space: nowrap; }
        [data-testid="stDataFrame"] { font-size: .85rem; }
    }

    @media (max-width: 480px) {
        .st-key-top_navbar .brand-logo { width: 40px; height: 40px; border-radius: 12px; }
        .st-key-top_navbar .brand-title { font-size: .94rem; }
        .st-key-top_navbar .brand-subtitle { display: none !important; }
        .metric-grid { grid-template-columns: 1fr; }
        .metric-card { min-height: 96px; }
        .progress-meta { font-size: .77rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_runtime_ui_css() -> None:
    """Apply session-controlled light/dark theme and mobile-menu visibility."""
    theme = st.session_state.get("ui_theme", "dark")
    menu_open = bool(st.session_state.get("mobile_menu_open", False))

    if theme == "light":
        st.markdown(
            """
            <style>
            /* v8 light palette: every interactive surface gets its own contrast-safe color. */
            :root {
                color-scheme: light;
                --ink-950: #eef5ff;
                --ink-900: #f4f8ff;
                --ink-850: #e9f1fb;
                --ink-800: #dce8f6;
                --ink-750: #cadbed;
                --line: #bfd2e8;
                --line-strong: #8fb1d4;
                --text: #0b1f36;
                --muted: #55708d;
                --success: #13855a;
                --danger: #c13e52;
                --shadow: 0 18px 44px rgba(34,76,124,.16);
            }

            html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
                color: #0b1f36 !important;
                background: #eef5ff !important;
            }
            .stApp {
                background:
                    radial-gradient(circle at 90% -8%, rgba(47,128,255,.19), transparent 30rem),
                    radial-gradient(circle at 6% 20%, rgba(37,214,255,.09), transparent 25rem),
                    linear-gradient(145deg, #fafdff 0%, #edf5ff 48%, #f7fbff 100%) !important;
            }
            h1, h2, h3, h4, h5, h6, p, label, .stCaption, [data-testid="stMarkdownContainer"] {
                color: #0b1f36 !important;
            }
            a { color: #135fca !important; }

            /* Custom header/navigation surfaces. */
            .st-key-top_navbar {
                background:
                    radial-gradient(circle at 12% 0%, rgba(47,128,255,.12), transparent 15rem),
                    linear-gradient(120deg, #ffffff 0%, #edf5ff 52%, #f8fcff 100%) !important;
                border-color: #b4cce5 !important;
                box-shadow:
                    0 18px 42px rgba(36,78,126,.17),
                    0 4px 12px rgba(36,78,126,.08),
                    inset 0 1px 0 rgba(255,255,255,.94) !important;
            }
            .st-key-top_navbar::after { background: rgba(47,128,255,.10) !important; }
            .st-key-top_navbar .brand-title,
            .st-key-top_navbar .user-name { color: #0b2340 !important; }
            .st-key-top_navbar .brand-subtitle,
            .st-key-top_navbar .user-goal { color: #58738f !important; }

            .app-topbar {
                background: linear-gradient(115deg, #e9f2ff, #ffffff 42%, #eefbff) !important;
                border-color: #bfd2e7 !important;
                box-shadow: 0 14px 34px rgba(42,72,108,.11) !important;
            }
            .app-topbar::after { background: rgba(47,128,255,.08) !important; }
            .brand-title, .user-name, .page-title, .metric-value { color: #0a1c34 !important; }
            .brand-subtitle, .user-goal, .page-subtitle, .small-muted, .nav-caption { color: #5b718b !important; }
            .user-chip {
                background: #ffffff !important;
                border-color: #c5d6e7 !important;
                box-shadow: 0 7px 18px rgba(42,72,108,.07) !important;
            }
            .user-dot { background: #168a5b !important; box-shadow: 0 0 12px rgba(22,138,91,.25) !important; }

            .st-key-main_navigation [role="radiogroup"] {
                background: #ffffff !important;
                border-color: #c5d6e7 !important;
                box-shadow: 0 10px 26px rgba(42,72,108,.09) !important;
            }
            .st-key-main_navigation [role="radiogroup"] > label {
                background: transparent !important;
                border-color: transparent !important;
            }
            .st-key-main_navigation [role="radiogroup"] > label:hover {
                background: #eaf3ff !important;
                border-color: #bfd5ef !important;
            }
            .st-key-main_navigation p { color: #27445f !important; }
            .st-key-main_navigation [role="radiogroup"] > label:has(input:checked) {
                background: linear-gradient(135deg, #125cc9, #2f80ff 65%, #147ab8) !important;
                border-color: #2f80ff !important;
                box-shadow: 0 8px 22px rgba(47,128,255,.20) !important;
            }
            .st-key-main_navigation [role="radiogroup"] > label:has(input:checked) p { color: #ffffff !important; }

            /* Page cards and metrics. */
            .page-icon {
                background: #e8f2ff !important;
                border-color: #c0d6ef !important;
                color: #1768d8 !important;
            }
            .metric-card {
                background: linear-gradient(150deg, #ffffff 0%, #f0f6ff 100%) !important;
                border-color: #bdd2e8 !important;
                box-shadow: 0 16px 34px rgba(37,78,124,.14), 0 3px 8px rgba(37,78,124,.06) !important;
            }
            .metric-label { color: #32669f !important; }
            .metric-hint { color: #687e97 !important; }
            .progress-panel, .soft-card {
                background: #ffffff !important;
                border-color: #c9d8e8 !important;
                color: #18324f !important;
                box-shadow: 0 14px 30px rgba(37,78,124,.12), 0 2px 7px rgba(37,78,124,.05) !important;
            }
            .hero-card {
                background: linear-gradient(125deg, #e9f3ff, #ffffff 58%, #eefbff) !important;
                border-color: #bfd4e9 !important;
                border-right-color: #2f80ff !important;
                color: #18324f !important;
                box-shadow: 0 10px 24px rgba(42,72,108,.07) !important;
            }
            .progress-meta { color: #45627f !important; }
            .progress-track { background: #dce7f2 !important; border-color: #cbd9e8 !important; }
            .progress-fill { box-shadow: 0 0 12px rgba(47,128,255,.20) !important; }

            div[data-testid="stMetric"] {
                background: #ffffff !important;
                border-color: #c9d8e8 !important;
                box-shadow: 0 9px 22px rgba(42,72,108,.07) !important;
            }
            div[data-testid="stMetric"] label { color: #32669f !important; }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0a1c34 !important; }
            div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #365a7d !important; }

            /* Secondary/neutral buttons: visible on white background. */
            .stButton > button,
            .stFormSubmitButton > button,
            .stDownloadButton > button,
            button[kind="secondary"] {
                background: linear-gradient(180deg, #f9fcff, #e8f1fb) !important;
                color: #123c69 !important;
                border: 1px solid #a9c4df !important;
                box-shadow: 0 5px 14px rgba(36,75,115,.08) !important;
            }
            .stButton > button:hover,
            .stFormSubmitButton > button:hover,
            .stDownloadButton > button:hover,
            button[kind="secondary"]:hover {
                background: #dcecff !important;
                color: #0b3159 !important;
                border-color: #6da2d7 !important;
                box-shadow: 0 8px 18px rgba(47,128,255,.14) !important;
            }
            .stButton > button:active,
            .stFormSubmitButton > button:active,
            .stDownloadButton > button:active {
                background: #cfe4fb !important;
                color: #082a4d !important;
            }

            /* Primary actions stay saturated blue in light mode. */
            button[kind="primary"],
            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                background: linear-gradient(100deg, #0e5cc9, #2f80ff 58%, #0877ad) !important;
                color: #ffffff !important;
                border-color: #176bd5 !important;
                box-shadow: 0 9px 22px rgba(47,128,255,.24) !important;
            }
            button[kind="primary"]:hover,
            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {
                background: linear-gradient(100deg, #0a4fae, #1f70e7 58%, #056a9d) !important;
                color: #ffffff !important;
                border-color: #0a58b9 !important;
            }

            /* Header action buttons. */
            .st-key-mobile_menu_toggle button {
                background: #e7f1ff !important;
                color: #104f94 !important;
                border-color: #a9c6e4 !important;
            }
            .st-key-mobile_menu_toggle button:hover {
                background: #d7eaff !important;
                color: #0b447f !important;
                border-color: #7fa9d3 !important;
            }
            .st-key-theme_toggle button {
                background: linear-gradient(145deg, #ffffff, #fff6d9) !important;
                color: #b46a00 !important;
                border-color: #e3c77c !important;
                box-shadow: 0 9px 22px rgba(167,116,18,.16), 0 2px 6px rgba(36,75,115,.08) !important;
            }
            .st-key-theme_toggle button:hover {
                background: linear-gradient(145deg, #fffdf5, #ffedb2) !important;
                color: #9a5700 !important;
                border-color: #d9b653 !important;
                box-shadow: 0 12px 26px rgba(167,116,18,.22) !important;
                transform: translateY(-1px);
            }
            .st-key-logout_top button {
                background: #fff0f2 !important;
                color: #a72f43 !important;
                border-color: #e4aeba !important;
            }
            .st-key-logout_top button:hover {
                background: #ffe2e7 !important;
                color: #8e2235 !important;
                border-color: #d98d9d !important;
            }

            /* Text/number/date/time inputs and text areas. */
            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            div[data-baseweb="textarea"],
            textarea,
            [data-testid="stDateInput"] > div > div,
            [data-testid="stTimeInput"] > div > div {
                background: #ffffff !important;
                border-color: #b9cde1 !important;
                color: #0d1b2e !important;
            }
            div[data-baseweb="input"] > div:focus-within,
            div[data-baseweb="select"] > div:focus-within,
            div[data-baseweb="textarea"]:focus-within {
                border-color: #2f80ff !important;
                box-shadow: 0 0 0 1px rgba(47,128,255,.20) !important;
            }
            input, textarea, [data-baseweb="select"] input {
                color: #0d1b2e !important;
                caret-color: #1768d8 !important;
                -webkit-text-fill-color: #0d1b2e !important;
            }
            input::placeholder, textarea::placeholder { color: #788ca2 !important; opacity: 1 !important; }

            /* v9: text fields are explicitly styled so typed food names never disappear. */
            [data-testid="stTextInput"] [data-baseweb="input"],
            [data-testid="stTextInput"] [data-baseweb="input"] > div {
                background: #ffffff !important;
                border-color: #9fbddb !important;
                box-shadow: inset 0 1px 2px rgba(28,61,98,.04), 0 3px 9px rgba(28,61,98,.055) !important;
            }
            [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
            [data-testid="stTextInput"] [data-baseweb="input"] > div:focus-within {
                border-color: #2f80ff !important;
                box-shadow: 0 0 0 3px rgba(47,128,255,.14), 0 6px 16px rgba(47,128,255,.09) !important;
            }
            [data-testid="stTextInput"] input {
                background: transparent !important;
                color: #071b31 !important;
                -webkit-text-fill-color: #071b31 !important;
                caret-color: #1267d8 !important;
                opacity: 1 !important;
                font-weight: 700 !important;
                text-shadow: none !important;
            }
            [data-testid="stTextInput"] input::placeholder {
                color: #6d839a !important;
                -webkit-text-fill-color: #6d839a !important;
                opacity: 1 !important;
                font-weight: 500 !important;
            }
            [data-testid="stTextInput"] label p {
                color: #173b61 !important;
                font-weight: 750 !important;
            }

            /* v10: search controls stay navy in Light Mode, so typed text must stay bright. */
            [class*="st-key-meal_quick_food_search_"] [data-baseweb="input"],
            [class*="st-key-meal_quick_food_search_"] [data-baseweb="input"] > div,
            [class*="st-key-meal_food_search_"] [data-baseweb="input"],
            [class*="st-key-meal_food_search_"] [data-baseweb="input"] > div {
                background: linear-gradient(180deg, #173a61 0%, #102b49 100%) !important;
                border-color: #4b88c8 !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 7px 18px rgba(23,58,97,.16) !important;
            }
            [class*="st-key-meal_quick_food_search_"] [data-baseweb="input"]:focus-within,
            [class*="st-key-meal_quick_food_search_"] [data-baseweb="input"] > div:focus-within,
            [class*="st-key-meal_food_search_"] [data-baseweb="input"]:focus-within,
            [class*="st-key-meal_food_search_"] [data-baseweb="input"] > div:focus-within {
                border-color: #63a8ff !important;
                box-shadow: 0 0 0 3px rgba(47,128,255,.20), 0 9px 22px rgba(23,58,97,.20) !important;
            }
            [class*="st-key-meal_quick_food_search_"] input,
            [class*="st-key-meal_food_search_"] input {
                color: #f7fbff !important;
                -webkit-text-fill-color: #f7fbff !important;
                caret-color: #dcecff !important;
                font-weight: 700 !important;
            }
            [class*="st-key-meal_quick_food_search_"] input::placeholder,
            [class*="st-key-meal_food_search_"] input::placeholder {
                color: #b9d2ec !important;
                -webkit-text-fill-color: #b9d2ec !important;
                opacity: 1 !important;
                font-weight: 500 !important;
            }

            /* Search-result selectors follow the same navy/light-text treatment. */
            [class*="st-key-meal_quick_food_result_"] [data-baseweb="select"] > div,
            [class*="st-key-meal_food_search_result_"] [data-baseweb="select"] > div {
                background: linear-gradient(180deg, #173a61 0%, #102b49 100%) !important;
                border-color: #4b88c8 !important;
                box-shadow: 0 6px 16px rgba(23,58,97,.13) !important;
            }
            [class*="st-key-meal_quick_food_result_"] [data-baseweb="select"] span,
            [class*="st-key-meal_quick_food_result_"] [data-baseweb="select"] div,
            [class*="st-key-meal_food_search_result_"] [data-baseweb="select"] span,
            [class*="st-key-meal_food_search_result_"] [data-baseweb="select"] div {
                color: #f7fbff !important;
                -webkit-text-fill-color: #f7fbff !important;
            }
            [class*="st-key-meal_quick_food_result_"] [data-baseweb="select"] svg,
            [class*="st-key-meal_food_search_result_"] [data-baseweb="select"] svg {
                fill: #dcecff !important;
                color: #dcecff !important;
            }

            div[data-baseweb="select"] svg,
            div[data-baseweb="input"] svg { fill: #456987 !important; color: #456987 !important; }

            /* Number input stepper buttons must not disappear. */
            [data-testid="stNumberInput"] button {
                background: #e8f1fb !important;
                color: #123c69 !important;
                border-color: #b9cde1 !important;
            }
            [data-testid="stNumberInput"] button:hover {
                background: #d7e9fb !important;
                color: #0b3159 !important;
            }
            [data-testid="stNumberInput"] button svg { fill: #123c69 !important; color: #123c69 !important; }

            /* Dropdowns, multiselect tokens and popovers. */
            [data-baseweb="popover"] > div,
            [role="listbox"] {
                background: #ffffff !important;
                border: 1px solid #b8cce0 !important;
                color: #0d1b2e !important;
                box-shadow: 0 12px 30px rgba(42,72,108,.14) !important;
            }
            [role="option"] { color: #0d1b2e !important; background: #ffffff !important; }
            [role="option"]:hover, [role="option"][aria-selected="true"] {
                background: #e5f1ff !important;
                color: #0a3564 !important;
            }
            [data-baseweb="tag"] {
                background: #dfeeff !important;
                color: #123c69 !important;
                border-color: #b4cee8 !important;
            }
            [data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #123c69 !important; fill: #123c69 !important; }

            /* File uploader and its browse button. */
            [data-testid="stFileUploader"] {
                background: #ffffff !important;
                border-color: #a9c6e4 !important;
                color: #173652 !important;
            }
            [data-testid="stFileUploader"] section {
                background: #f5f9fe !important;
                border-color: #b8cee4 !important;
            }
            [data-testid="stFileUploader"] button {
                background: #e5f0fc !important;
                color: #123c69 !important;
                border-color: #a9c4df !important;
            }

            /* Tabs and radio/checkbox controls. */
            [data-baseweb="tab-list"] { border-bottom-color: #c7d7e7 !important; }
            [data-baseweb="tab"] {
                background: #edf4fb !important;
                color: #35536f !important;
            }
            [data-baseweb="tab"][aria-selected="true"] {
                background: #dcecff !important;
                color: #0b4f9a !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] > label {
                background: #ffffff !important;
                border-color: #c2d3e4 !important;
                color: #18324f !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] > label:hover { background: #eaf3ff !important; }
            [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p { color: #18324f !important; }

            /* Alerts are explicitly recolored per semantic type. */
            [data-testid="stAlert"] {
                color: #17324f !important;
                border-color: #bfd0e1 !important;
            }
            [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: inherit !important; }
            [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { background: #e8f3ff !important; }
            [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { background: #e9f8f1 !important; color: #135c40 !important; }
            [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { background: #fff7df !important; color: #6e5310 !important; }
            [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { background: #fff0f2 !important; color: #8e2d3c !important; }

            /* Tables/dataframes and separators. */
            [data-testid="stDataFrame"] {
                border-color: #c9d8e8 !important;
                box-shadow: 0 10px 24px rgba(42,72,108,.08) !important;
                background: #ffffff !important;
            }
            hr { border-color: #cbd9e7 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    mobile_display = "block" if menu_open else "none"
    menu_css = """
    <style>
    @media (max-width: 820px) {
        .st-key-main_navigation { display: __MOBILE_DISPLAY__ !important; }
        .st-key-main_navigation [role="radiogroup"] { display: grid !important; }
    }
    </style>
    """.replace("__MOBILE_DISPLAY__", mobile_display)
    st.markdown(menu_css, unsafe_allow_html=True)


render_runtime_ui_css()

init_db()

MEAL_TYPE_FA = {
    "breakfast": "صبحانه",
    "morning_snack": "میان‌وعده صبح",
    "lunch": "ناهار",
    "afternoon_snack": "عصرانه",
    "dinner": "شام",
    "evening_snack": "میان‌وعده شب",
    "treat": "خوراکی / شیرینی",
    "other": "سایر",
    "snack": "میان‌وعده",  # نمایش داده‌های نسخه قبلی
}
MEAL_ENTRY_TYPES = [
    "breakfast",
    "morning_snack",
    "lunch",
    "afternoon_snack",
    "dinner",
    "evening_snack",
    "treat",
    "other",
]
MEAL_TYPE_BY_FA = {MEAL_TYPE_FA[key]: key for key in MEAL_ENTRY_TYPES}
WEEKDAY_FA = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه"]


@st.cache_resource(show_spinner=False)
def get_classifier() -> HybridFoodClassifier:
    # Food-101 is the general-food expert and CLIP is the Iranian-food specialist/refiner.
    # Streamlit keeps both models and the Iranian CLIP text embeddings in memory.
    return HybridFoodClassifier()


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("فایل انتخاب‌شده یک تصویر معتبر نیست.") from exc


def confidence_for_label(predictions: list[HybridPrediction], label: str) -> float:
    for prediction in predictions:
        if prediction.label == label:
            return prediction.score
    return 0.0


def format_kcal(value: float) -> str:
    return f"{value:,.0f} kcal"


def toggle_ui_theme() -> None:
    current = st.session_state.get("ui_theme", "dark")
    st.session_state["ui_theme"] = "light" if current == "dark" else "dark"


def toggle_mobile_menu() -> None:
    st.session_state["mobile_menu_open"] = not bool(
        st.session_state.get("mobile_menu_open", False)
    )


def close_mobile_menu() -> None:
    st.session_state["mobile_menu_open"] = False


NAV_PAGES = [
    "داشبورد امروز",
    "ثبت وعده جدید",
    "تاریخچه",
    "گزارش هفتگی",
    "پروفایل و تنظیمات",
]


def render_top_navigation(user: dict[str, Any]) -> str:
    """Render the custom in-page navbar and keep theme control inside it."""
    username = escape(str(user.get("username") or "کاربر"))
    goal = float(user.get("daily_calorie_goal") or 0.0)
    menu_open = bool(st.session_state.get("mobile_menu_open", False))
    current_theme = st.session_state.get("ui_theme", "dark")

    with st.container(key="top_navbar"):
        brand_col, user_col, theme_col = st.columns([5.3, 2.1, 0.55])
        with brand_col:
            st.markdown(
                """
                <div class="brand-wrap">
                  <div class="brand-logo">🍽️</div>
                  <div>
                    <div class="brand-title">دفترچه هوشمند تغذیه</div>
                    <div class="brand-subtitle">تشخیص غذا، ثبت کالری و گزارش هفتگی در یک داشبورد</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with user_col:
            st.markdown(
                f"""
                <div class="user-chip">
                  <span class="user-dot"></span>
                  <div>
                    <div class="user-name">{username}</div>
                    <div class="user-goal">هدف روزانه: <span dir="ltr">{goal:,.0f} kcal</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with theme_col:
            st.button(
                "☀️" if current_theme == "light" else "🌙",
                use_container_width=True,
                key="theme_toggle",
                on_click=toggle_ui_theme,
                help="تغییر تم نمایش",
            )

    menu_col, spacer_col, logout_col = st.columns([1.15, 5.0, 1.4])
    with menu_col:
        st.button(
            "✕ بستن" if menu_open else "☰ منو",
            use_container_width=True,
            key="mobile_menu_toggle",
            on_click=toggle_mobile_menu,
        )
    with spacer_col:
        st.empty()
    with logout_col:
        if st.button("خروج از حساب", use_container_width=True, key="logout_top"):
            clear_meal_builder(reset_form=True)
            st.session_state.pop("authenticated_user_id", None)
            st.session_state["mobile_menu_open"] = False
            st.session_state["flash_message"] = ("info", "از حساب خارج شدید.")
            st.rerun()

    page = st.radio(
        "بخش‌های برنامه",
        NAV_PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
        on_change=close_mobile_menu,
    )
    return page


def render_page_header(title: str, subtitle: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-title-row">
            <div class="page-icon">{escape(icon)}</div>
            <div>
              <div class="page-title">{escape(title)}</div>
              <div class="page-subtitle">{escape(subtitle)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_metrics(
    *, consumed: float, remaining: float, meal_count: int, item_count: int, goal: float
) -> None:
    remaining_title = "باقی‌مانده" if remaining >= 0 else "بیش از هدف"
    progress_percent = 0.0 if goal <= 0 else min(max(consumed / goal * 100.0, 0.0), 100.0)
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">مصرف امروز</div>
            <div class="metric-value">{consumed:,.0f} kcal</div>
            <div class="metric-hint">کالری ثبت‌شده امروز</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">{remaining_title}</div>
            <div class="metric-value">{abs(remaining):,.0f} kcal</div>
            <div class="metric-hint">بر اساس هدف روزانه</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">تعداد وعده‌ها</div>
            <div class="metric-value">{int(meal_count):,}</div>
            <div class="metric-hint">وعده‌های ثبت‌شده امروز</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">تعداد اقلام</div>
            <div class="metric-value">{int(item_count):,}</div>
            <div class="metric-hint">غذاها و خوراکی‌های امروز</div>
          </div>
        </div>
        <div class="progress-panel">
          <div class="progress-meta">
            <span>پیشرفت هدف روزانه</span>
            <span dir="ltr">{consumed:,.0f} / {goal:,.0f} kcal</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width:{progress_percent:.1f}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def default_meal_type() -> str:
    """Suggest a category from local clock time while keeping it editable."""
    hour = datetime.now().hour
    if 5 <= hour < 10:
        return "breakfast"
    if 10 <= hour < 12:
        return "morning_snack"
    if 12 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "afternoon_snack"
    if 18 <= hour < 22:
        return "dinner"
    return "evening_snack"


def meal_type_select_index(meal_type: str | None = None) -> int:
    selected = meal_type if meal_type in MEAL_ENTRY_TYPES else default_meal_type()
    return MEAL_ENTRY_TYPES.index(selected)


def clean_time_for_display(value: str | None) -> str:
    return value or "—"


def meal_builder_key(name: str) -> str:
    version = int(st.session_state.get("meal_form_version", 0))
    return f"meal_{name}_{version}"


def clear_meal_builder(*, reset_form: bool = False) -> None:
    fixed_keys = (
        "meal_predictions",
        "meal_selected_foods",
        "meal_nutrition_by_label",
        "meal_image_hash",
        "meal_uploaded_hash",
        "quick_catalog_label",
        "quick_catalog_nutrition",
    )
    for key in fixed_keys:
        st.session_state.pop(key, None)

    if reset_form:
        # A new key namespace resets every widget on the next rerun without
        # mutating widget state after it has already been instantiated.
        version = int(st.session_state.get("meal_form_version", 0))
        st.session_state["meal_form_version"] = version + 1


def show_flash() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not flash:
        return
    level, message = flash
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def current_user() -> dict[str, Any] | None:
    user_id = st.session_state.get("authenticated_user_id")
    if user_id is None:
        return None
    user = get_user_by_id(int(user_id))
    if user is None:
        st.session_state.pop("authenticated_user_id", None)
    return user


def render_authentication() -> None:
    render_page_header("دفترچه هوشمند تغذیه", "ورود، ساخت حساب و مدیریت دفترچه تغذیه شخصی", "🍽️")
    st.markdown(
        """
        <div class="hero-card">
          غذای روزانه‌تان را ثبت کنید، مصرف کالری را ببینید و در پایان هفته
          یک گزارش روشن از الگوی تغذیه خود دریافت کنید.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.7, 1])
    with center:
        login_tab, register_tab, reset_tab = st.tabs(
            ["ورود", "ساخت حساب", "فراموشی رمز"]
        )

        with login_tab:
            with st.form("login_form"):
                identifier = st.text_input("نام کاربری یا ایمیل")
                password = st.text_input("رمز عبور", type="password")
                login_submitted = st.form_submit_button(
                    "ورود به حساب", type="primary", use_container_width=True
                )

            if login_submitted:
                account = get_user_for_login(identifier)
                if account and verify_password(password, account["password_hash"]):
                    if not bool(account.get("email_verified", 0)):
                        st.error(
                            "ایمیل این حساب هنوز تأیید نشده است. از تب «ساخت حساب» کد تأیید را وارد کنید."
                        )
                    else:
                        st.session_state["authenticated_user_id"] = int(account["id"])
                        st.session_state["flash_message"] = (
                            "success",
                            f"خوش آمدید {account['username']}.",
                        )
                        st.rerun()
                else:
                    st.error("نام کاربری/ایمیل یا رمز عبور درست نیست.")

        with register_tab:
            st.caption("پس از ثبت اطلاعات، یک کد ۶ رقمی واقعاً به ایمیل شما ارسال می‌شود.")
            with st.form("registration_form"):
                username = st.text_input("نام کاربری", key="register_username")
                email = st.text_input("ایمیل *", key="register_email")
                goal = st.number_input(
                    "هدف کالری روزانه",
                    min_value=500.0,
                    max_value=10000.0,
                    value=2000.0,
                    step=50.0,
                )
                password = st.text_input(
                    "رمز عبور (حداقل ۸ نویسه)",
                    type="password",
                    key="register_password",
                )
                confirmation = st.text_input(
                    "تکرار رمز عبور",
                    type="password",
                    key="register_confirmation",
                )
                register_submitted = st.form_submit_button(
                    "ساخت حساب", type="primary", use_container_width=True
                )

            if register_submitted:
                try:
                    if not smtp_is_configured():
                        raise AccountValidationError(
                            "ارسال ایمیل هنوز تنظیم نشده است. ابتدا SMTP و App Password را در فایل .env وارد کنید."
                        )
                    clean_username = validate_username(username)
                    clean_email = validate_email(email)
                    validate_password(password)
                    if password != confirmation:
                        raise AccountValidationError("تکرار رمز عبور یکسان نیست.")
                    user_id = create_user(
                        username=clean_username,
                        email=clean_email,
                        password_hash=hash_password(password),
                        daily_calorie_goal=float(goal),
                        email_verified=False,
                    )
                    code = generate_reset_code()
                    create_email_verification_token(
                        user_id,
                        hash_reset_code(code),
                        expires_minutes=15,
                    )
                    if not send_email_verification_code(
                        clean_email,
                        code,
                        expires_minutes=15,
                    ):
                        raise AccountValidationError(
                            "اتصال SMTP کامل نیست و کد تأیید ارسال نشد."
                        )
                except (AccountValidationError, ValueError) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error(
                        "حساب ساخته شد اما ارسال ایمیل ناموفق بود. تنظیمات Gmail/App Password را بررسی و از بخش «ارسال مجدد» استفاده کنید."
                    )
                else:
                    st.session_state["registration_verification_email"] = clean_email
                    st.success(
                        "حساب اولیه ساخته شد و کد تأیید به ایمیل ارسال شد. کد را در بخش زیر وارد کنید."
                    )

            st.divider()
            st.markdown("#### تأیید ایمیل")
            st.caption("کد تا ۱۵ دقیقه معتبر است و پس از ۵ تلاش اشتباه غیرفعال می‌شود.")
            with st.form("verify_registration_email_form"):
                verify_email = st.text_input(
                    "ایمیل",
                    value=str(st.session_state.get("registration_verification_email", "")),
                    key="register_verify_email",
                )
                verify_code = st.text_input(
                    "کد ۶ رقمی تأیید",
                    max_chars=10,
                    key="register_verify_code",
                )
                verify_submitted = st.form_submit_button(
                    "تأیید ایمیل", type="primary", use_container_width=True
                )

            if verify_submitted:
                try:
                    clean_email = validate_email(verify_email)
                    verification = get_active_email_verification(clean_email)
                    if verification is None:
                        raise AccountValidationError(
                            "کد معتبر پیدا نشد؛ از گزینه ارسال مجدد، کد جدید بگیرید."
                        )
                    if not verify_reset_code(verify_code, verification["token_hash"]):
                        record_email_verification_failure(int(verification["id"]))
                        raise AccountValidationError("کد تأیید نادرست است.")
                    if not complete_email_verification(
                        int(verification["id"]), int(verification["user_id"])
                    ):
                        raise AccountValidationError(
                            "کد منقضی یا مصرف شده است؛ کد جدید درخواست کنید."
                        )
                except (AccountValidationError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("registration_verification_email", None)
                    st.session_state["authenticated_user_id"] = int(verification["user_id"])
                    st.session_state["flash_message"] = (
                        "success",
                        "ایمیل تأیید شد و حساب شما فعال شد.",
                    )
                    st.rerun()

            with st.form("resend_registration_code_form"):
                resend_email = st.text_input(
                    "ارسال مجدد به ایمیل",
                    value=str(st.session_state.get("registration_verification_email", "")),
                    key="register_resend_email",
                )
                resend_submitted = st.form_submit_button(
                    "ارسال مجدد کد", use_container_width=True
                )

            if resend_submitted:
                try:
                    if not smtp_is_configured():
                        raise AccountValidationError("SMTP هنوز کامل تنظیم نشده است.")
                    clean_email = validate_email(resend_email)
                    account = get_user_by_email(clean_email)
                    if account is None or bool(account.get("email_verified", 0)):
                        raise AccountValidationError(
                            "حساب تأییدنشده‌ای با این ایمیل پیدا نشد."
                        )
                    code = generate_reset_code()
                    create_email_verification_token(
                        int(account["id"]),
                        hash_reset_code(code),
                        expires_minutes=15,
                    )
                    if not send_email_verification_code(clean_email, code, expires_minutes=15):
                        raise AccountValidationError("ارسال ایمیل ناموفق بود.")
                except (AccountValidationError, ValueError) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("ارسال ایمیل ناموفق بود؛ App Password و اتصال اینترنت را بررسی کنید.")
                else:
                    st.session_state["registration_verification_email"] = clean_email
                    st.success("کد تأیید جدید به ایمیل ارسال شد.")

        with reset_tab:
            st.markdown("#### دریافت کد بازیابی")
            st.caption(
                "ایمیل حساب را وارد کنید. کد یک‌بارمصرف تا ۱۵ دقیقه معتبر است."
            )
            with st.form("request_password_reset_form"):
                reset_email_request = st.text_input(
                    "ایمیل حساب",
                    key="reset_email_request",
                )
                request_submitted = st.form_submit_button(
                    "ارسال کد بازیابی", type="primary", use_container_width=True
                )

            if request_submitted:
                try:
                    clean_email = validate_email(reset_email_request)
                except AccountValidationError as exc:
                    st.error(str(exc))
                else:
                    account = get_user_by_email(clean_email)
                    delivery_succeeded = False
                    debug_code: str | None = None
                    if account is not None:
                        code = generate_reset_code()
                        try:
                            create_password_reset_token(
                                int(account["id"]),
                                hash_reset_code(code),
                                expires_minutes=15,
                            )
                            delivery_succeeded = send_password_reset_code(
                                clean_email,
                                code,
                                expires_minutes=15,
                            )
                            if not delivery_succeeded and reset_code_debug_enabled():
                                debug_code = code
                        except ValueError:
                            # Keep the public response neutral to avoid revealing accounts.
                            pass
                        except Exception:
                            # SMTP failures must not expose credentials or account existence.
                            if reset_code_debug_enabled():
                                debug_code = code

                    st.session_state["password_reset_email"] = clean_email
                    st.success(
                        "اگر این ایمیل در سیستم ثبت شده باشد، کد بازیابی برای آن صادر شده است."
                    )
                    if debug_code:
                        st.warning(
                            "SMTP تنظیم نشده یا ارسال ناموفق بود. حالت آزمایشی محلی فعال است."
                        )
                        st.code(debug_code, language=None)
                        st.caption(
                            "برای انتشار عمومی RESET_CODE_DEBUG=false قرار دهید و SMTP را تنظیم کنید."
                        )
                    elif not smtp_is_configured():
                        st.info(
                            "SMTP هنوز تنظیم نشده است. در حالت عمومی، مدیر برنامه باید تنظیمات ایمیل را تکمیل کند."
                        )

            st.divider()
            st.markdown("#### تعیین رمز جدید")
            with st.form("complete_password_reset_form"):
                reset_email = st.text_input(
                    "ایمیل",
                    value=str(st.session_state.get("password_reset_email", "")),
                    key="reset_email_complete",
                )
                reset_code = st.text_input(
                    "کد ۶ رقمی",
                    max_chars=10,
                    key="reset_code_complete",
                )
                new_password = st.text_input(
                    "رمز عبور جدید (حداقل ۸ نویسه)",
                    type="password",
                    key="reset_new_password",
                )
                reset_confirmation = st.text_input(
                    "تکرار رمز عبور جدید",
                    type="password",
                    key="reset_new_password_confirmation",
                )
                complete_submitted = st.form_submit_button(
                    "ثبت رمز عبور جدید", type="primary", use_container_width=True
                )

            if complete_submitted:
                try:
                    clean_email = validate_email(reset_email)
                    validate_password(new_password)
                    if new_password != reset_confirmation:
                        raise AccountValidationError("تکرار رمز عبور جدید یکسان نیست.")
                    reset_record = get_active_password_reset(clean_email)
                    if reset_record is None:
                        raise AccountValidationError(
                            "کد معتبر پیدا نشد؛ کد جدید درخواست کنید."
                        )
                    if not verify_reset_code(reset_code, reset_record["token_hash"]):
                        record_password_reset_failure(int(reset_record["id"]))
                        raise AccountValidationError("کد بازیابی نادرست است.")
                    if not complete_password_reset(
                        int(reset_record["id"]),
                        int(reset_record["user_id"]),
                        hash_password(new_password),
                    ):
                        raise AccountValidationError(
                            "کد منقضی یا مصرف شده است؛ کد جدید درخواست کنید."
                        )
                except (AccountValidationError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("password_reset_email", None)
                    st.success("رمز عبور تغییر کرد. اکنون از تب ورود وارد شوید.")

    st.caption(
        "اطلاعات حساب و وعده‌ها در SQLite ذخیره می‌شوند؛ کدهای بازیابی به‌صورت هش‌شده نگهداری می‌شوند."
    )



def meal_summary_table(meals: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "تاریخ": [item["meal_date"] for item in meals],
            "ساعت": [clean_time_for_display(item.get("meal_time")) for item in meals],
            "وعده": [MEAL_TYPE_FA.get(item["meal_type"], item["meal_type"]) for item in meals],
            "تعداد اقلام": [int(item["item_count"]) for item in meals],
            "کالری": [round(float(item["total_calories"]), 1) for item in meals],
            "یادداشت": [item.get("notes") or "—" for item in meals],
        }
    )


def render_dashboard(user: dict[str, Any]) -> None:
    today = date.today()
    summary = daily_summary(int(user["id"]), today)
    goal = float(user["daily_calorie_goal"])
    consumed = float(summary["total_calories"])
    remaining = goal - consumed

    render_page_header(
        "داشبورد امروز",
        f"وضعیت ثبت تغذیه در {today.isoformat()}",
        "⌂",
    )
    render_dashboard_metrics(
        consumed=consumed,
        remaining=remaining,
        meal_count=int(summary["meal_count"]),
        item_count=int(summary["item_count"]),
        goal=goal,
    )
    if consumed > goal:
        st.warning(f"مصرف ثبت‌شده امروز {consumed - goal:,.0f} کیلوکالری بیشتر از هدف است.")

    st.subheader("وعده‌های امروز")
    meals = get_meals_for_date(int(user["id"]), today)
    if not meals:
        st.info("هنوز وعده‌ای برای امروز ثبت نشده است. از بخش «ثبت وعده جدید» شروع کنید.")
    else:
        st.dataframe(meal_summary_table(meals), hide_index=True, use_container_width=True)
        for meal in meals:
            meal_clock = f" ساعت {meal['meal_time']}" if meal.get("meal_time") else ""
            title = (
                f"{MEAL_TYPE_FA.get(meal['meal_type'], meal['meal_type'])}{meal_clock} — "
                f"{float(meal['total_calories']):,.0f} kcal"
            )
            with st.expander(title):
                items = get_meal_items(int(user["id"]), int(meal["id"]))
                st.dataframe(
                    pd.DataFrame(
                        {
                            "غذا": [item["food_name_fa"] for item in items],
                            "مقدار مصرف": [
                                f"{float(item['weight_grams']):,.0f} {short_unit_fa(item.get('measure_unit', 'g'))}"
                                for item in items
                            ],
                            "کالری": [round(float(item["estimated_calories"]), 1) for item in items],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if meal.get("notes"):
                    st.caption(f"یادداشت: {meal['notes']}")


def render_quick_custom_entry(user: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="soft-card">
          برای شیرینی، نوشیدنی یا خوراکی‌ای که در فهرست نیست، نام و کالری را مستقیم وارد کنید.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔎 انتخاب سریع از فهرست")
    quick_search = st.text_input(
        "جست‌وجوی غذا یا نوشیدنی",
        placeholder="مثلاً برنج، سالمون، نوشابه، قهوه یا pizza",
        key=meal_builder_key("quick_food_search"),
        help="جست‌وجو هم روی نام فارسی و هم نام انگلیسی انجام می‌شود.",
    )
    if quick_search.strip():
        quick_matches = search_food_labels(quick_search, limit=25)
        if quick_matches:
            quick_choice = st.selectbox(
                "نتایج جست‌وجو",
                options=quick_matches,
                format_func=to_persian,
                key=meal_builder_key("quick_food_result"),
            )
            if st.button(
                "استفاده از این خوراکی",
                use_container_width=True,
                key=meal_builder_key("quick_food_use"),
            ):
                try:
                    quick_nutrition = lookup_calories(quick_choice)
                except NutritionLookupError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["quick_catalog_label"] = quick_choice
                    st.session_state["quick_catalog_nutrition"] = quick_nutrition
                    st.session_state[meal_builder_key("custom_calculation_mode")] = (
                        "محاسبه بر اساس مقدار مصرف و کالری مرجع"
                    )
                    st.session_state[meal_builder_key("custom_kind")] = (
                        "نوشیدنی" if is_drink_label(quick_choice) else "غذا"
                    )
                    st.rerun()
        else:
            st.warning("موردی در فهرست پیدا نشد؛ می‌توانید نام خوراکی را پایین‌تر دستی وارد کنید.")

    quick_catalog_label = st.session_state.get("quick_catalog_label")
    quick_catalog_nutrition = st.session_state.get("quick_catalog_nutrition")
    quick_unit = measure_unit_for_label(quick_catalog_label) if quick_catalog_label else "g"
    quick_reference_kcal = (
        calories_for_display_unit(
            quick_catalog_label,
            quick_catalog_nutrition.calories_per_100g,
            source=quick_catalog_nutrition.source,
        )
        if quick_catalog_label and isinstance(quick_catalog_nutrition, NutritionResult)
        else None
    )
    if quick_catalog_label and isinstance(quick_catalog_nutrition, NutritionResult):
        st.success(
            f"انتخاب شده: {to_persian(quick_catalog_label)} — "
            f"{float(quick_reference_kcal):,.0f} kcal در ۱۰۰ {short_unit_fa(quick_unit)}"
        )

    custom_kind = st.radio(
        "نوع مورد",
        ["غذا", "نوشیدنی"],
        horizontal=True,
        key=meal_builder_key("custom_kind"),
        help="برای نوشیدنی‌ها مقدار مصرف بر حسب میلی‌لیتر و برای غذاها بر حسب گرم ثبت می‌شود.",
    )
    custom_unit = "ml" if custom_kind == "نوشیدنی" else "g"

    calculation_mode = st.radio(
        "روش واردکردن کالری",
        ["کالری کل خوراکی را می‌دانم", "محاسبه بر اساس مقدار مصرف و کالری مرجع"],
        horizontal=True,
        key=meal_builder_key("custom_calculation_mode"),
    )

    with st.form(meal_builder_key("quick_custom_form")):
        custom_name = st.text_input(
            "نام خوراکی",
            value=(to_persian(quick_catalog_label) if quick_catalog_label else ""),
            max_chars=100,
            placeholder="مثلاً یک عدد شیرینی دانمارکی",
        )
        meta1, meta2, meta3 = st.columns(3)
        with meta1:
            meal_type_fa = st.selectbox(
                "نوع وعده",
                options=[MEAL_TYPE_FA[key] for key in MEAL_ENTRY_TYPES],
                index=MEAL_ENTRY_TYPES.index("treat"),
            )
        with meta2:
            meal_date = st.date_input("تاریخ مصرف", value=date.today())
        with meta3:
            meal_time = st.time_input(
                "زمان مصرف",
                value=datetime.now().time().replace(second=0, microsecond=0),
                step=timedelta(minutes=5),
            )

        if calculation_mode == "کالری کل خوراکی را می‌دانم":
            calories_total = st.number_input(
                "کالری کل خوراکی",
                min_value=0.0,
                max_value=10000.0,
                value=150.0,
                step=10.0,
                help="مثلاً مقدار درج‌شده روی بسته‌بندی یا یک سهم مصرف‌شده.",
            )
            amount = st.number_input(
                f"{amount_label_fa(custom_unit)} (اختیاری برای ثبت دقیق‌تر)",
                min_value=1.0,
                max_value=max_amount(custom_unit),
                value=default_amount(custom_unit),
                step=1.0,
                format="%.0f",
                help=(
                    "برای نوشیدنی مثلاً ۲۵۰ میلی‌لیتر وارد کنید."
                    if custom_unit == "ml"
                    else "می‌توانید وزن را با دقت یک گرم، مثلاً ۱۸۲ گرم، وارد کنید."
                ),
            )
            calories_per_100g = float(calories_total) * 100.0 / float(amount)
        else:
            value1, value2 = st.columns(2)
            with value1:
                amount = st.number_input(
                    amount_label_fa(custom_unit),
                    min_value=1.0,
                    max_value=max_amount(custom_unit),
                    value=default_amount(custom_unit),
                    step=1.0,
                    format="%.0f",
                    help=(
                        "حجم نوشیدنی را با دقت یک میلی‌لیتر، مثلاً ۲۵۰ میلی‌لیتر، وارد کنید."
                        if custom_unit == "ml"
                        else "وزن غذا را با دقت یک گرم، مثلاً ۱۸۲ گرم، وارد کنید."
                    ),
                )
            with value2:
                calories_per_100g = st.number_input(
                    per_100_label_fa(custom_unit),
                    min_value=0.0,
                    max_value=3000.0,
                    value=(
                        float(quick_reference_kcal)
                        if quick_reference_kcal is not None and custom_unit == quick_unit
                        else (50.0 if custom_unit == "ml" else 300.0)
                    ),
                    step=1.0,
                    help=(
                        "برای نوشیدنی‌ها مقدار مرجع به ازای ۱۰۰ میلی‌لیتر ثبت می‌شود."
                        if custom_unit == "ml"
                        else "برای غذاها مقدار مرجع به ازای ۱۰۰ گرم ثبت می‌شود."
                    ),
                )
            calories_total = float(amount) * float(calories_per_100g) / 100.0

        notes = st.text_area(
            "یادداشت (اختیاری)",
            max_chars=500,
            placeholder="مثلاً همراه چای یا بعد از پیاده‌روی",
        )
        submitted = st.form_submit_button(
            "ثبت سریع خوراکی",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        name = custom_name.strip()
        if not name:
            st.error("نام خوراکی را وارد کنید.")
            return
        custom_label = "custom_" + hashlib.sha256(name.lower().encode("utf-8")).hexdigest()[:16]
        try:
            meal_id = create_meal(
                user_id=int(user["id"]),
                meal_type=MEAL_TYPE_BY_FA[meal_type_fa],
                meal_date=meal_date,
                meal_time=meal_time,
                items=[
                    {
                        "food_label": custom_label,
                        "food_name_fa": name,
                        "weight_grams": float(amount),
                        "measure_unit": custom_unit,
                        "calories_per_100g": float(calories_per_100g),
                        "confidence": 0.0,
                        "nutrition_description": "ثبت دستی توسط کاربر",
                        "source": "manual",
                    }
                ],
                notes=notes,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            clear_meal_builder(reset_form=True)
            st.session_state["flash_message"] = (
                "success",
                f"خوراکی شماره {meal_id} با {float(calories_total):,.0f} کیلوکالری ثبت شد.",
            )
            st.rerun()


def render_meal_entry(user: dict[str, Any]) -> None:
    render_page_header(
        "ثبت وعده جدید",
        "تصویر اختیاری است؛ تشخیص هوشمند، جست‌وجوی غذا و ثبت سریع در دسترس است.",
        "＋",
    )

    entry_method = st.radio(
        "روش ثبت",
        ["تصویر یا فهرست غذاها", "ثبت سریع خوراکی دلخواه"],
        horizontal=True,
        key=meal_builder_key("entry_method"),
    )
    if entry_method == "ثبت سریع خوراکی دلخواه":
        render_quick_custom_entry(user)
        return

    upload_key = meal_builder_key("image")
    selected_widget_key = meal_builder_key("selected_foods")
    uploaded_file = st.file_uploader(
        "تصویر غذا (اختیاری)",
        type=["jpg", "jpeg", "png", "webp"],
        key=upload_key,
        help="برای نتیجه بهتر، تصویر واضح و از نزدیک باشد.",
    )

    image: Image.Image | None = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("meal_uploaded_hash") != current_hash:
            # A changed image invalidates prior predictions and nutrition choices.
            st.session_state["meal_uploaded_hash"] = current_hash
            st.session_state.pop("meal_predictions", None)
            st.session_state.pop("meal_nutrition_by_label", None)
            st.session_state.pop("meal_selected_foods", None)
            st.session_state.pop(selected_widget_key, None)

        try:
            image = read_image(file_bytes)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.image(image, caption="تصویر ورودی", width=480)
            if st.button(
                "تشخیص هوشمند غذا (Food-101 + متخصص ایرانی)",
                type="primary",
                use_container_width=True,
                key=meal_builder_key("classify_button"),
            ):
                with st.spinner(
                    "ابتدا Food-101 نوع عمومی غذا را بررسی می‌کند. اگر خانواده واضحی مثل پاستا، برگر یا پیتزا تشخیص داده شود، "
                    "CLIP فقط داخل همان خانواده نتیجه را دقیق‌تر می‌کند؛ در غیر این صورت به‌عنوان متخصص غذاهای ایرانی فعال می‌شود. "
                    "اجرای اول به‌دلیل دانلود مدل دوم زمان بیشتری می‌برد..."
                ):
                    try:
                        predictions = get_classifier().predict(image, top_k=8)
                    except Exception as exc:
                        st.error(f"اجرای مدل ناموفق بود: {exc}")
                    else:
                        st.session_state["meal_predictions"] = predictions
                        st.session_state["meal_image_hash"] = current_hash
                        st.session_state["meal_selected_foods"] = [predictions[0].label]
                        st.session_state[selected_widget_key] = [predictions[0].label]
                        st.session_state["meal_nutrition_by_label"] = {}

    predictions: list[HybridPrediction] = st.session_state.get("meal_predictions", [])
    if predictions:
        best = predictions[0]
        st.markdown(
            f"""
            <div class="soft-card">
              <b>پیشنهاد نهایی:</b> {to_persian(best.label)}<br>
              <b>امتیاز تطبیق:</b> {best.score * 100:.1f}٪<br>
              <span style="opacity:.78">این عدد امتیاز رتبه‌بندی ترکیبی است و احتمال آماری کالیبره‌شده نیست.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("مشاهده جزئیات Food-101 و متخصص ایرانی"):
            source_fa = {
                "hybrid-confirmed": "Food-101 + تأیید CLIP",
                "iranian-zero-shot": "متخصص غذای ایرانی (CLIP)",
                "iranian-fine-grain": "تفکیک دقیق خورش‌های مشابه (CLIP)",
                "food-101": "Food-101",
                "family-confirmed": "Food-101 + تأیید در همان خانواده",
                "family-refiner": "اصلاح‌کننده همان خانواده (CLIP)",
            }
            st.dataframe(
                pd.DataFrame(
                    {
                        "غذا": [to_persian(item.label) for item in predictions],
                        "برچسب": [item.label.replace("_", " ") for item in predictions],
                        "امتیاز ترکیبی (٪)": [round(item.score * 100, 2) for item in predictions],
                        "Zero-Shot (٪)": [round(item.zero_shot_score * 100, 2) for item in predictions],
                        "Food-101 (٪)": [round(item.food101_confidence * 100, 2) for item in predictions],
                        "منبع": [source_fa.get(item.source, item.source) for item in predictions],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("#### 🔎 جست‌وجو و افزودن غذا")
    food_search = st.text_input(
        "نام غذا یا نوشیدنی را جست‌وجو کنید",
        placeholder="مثلاً برنج، سالمون، آب پرتقال یا hamburger",
        key=meal_builder_key("food_search"),
        help="برای نمایش نتیجه لازم نیست فهرست بزرگ را باز کنید؛ فارسی و انگلیسی قابل جست‌وجو است.",
    )
    if food_search.strip():
        search_matches = search_food_labels(food_search, limit=30)
        if search_matches:
            result_col, add_col = st.columns([4, 1])
            with result_col:
                search_choice = st.selectbox(
                    f"نتایج ({len(search_matches)} مورد)",
                    options=search_matches,
                    format_func=to_persian,
                    key=meal_builder_key("food_search_result"),
                )
            with add_col:
                st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
                if st.button(
                    "➕ افزودن",
                    use_container_width=True,
                    key=meal_builder_key("food_search_add"),
                ):
                    selected_now = list(st.session_state.get("meal_selected_foods", []))
                    if search_choice not in selected_now:
                        selected_now.append(search_choice)
                    st.session_state["meal_selected_foods"] = selected_now
                    st.session_state[selected_widget_key] = selected_now
                    st.rerun()
        else:
            st.warning("غذایی با این عبارت در فهرست پیدا نشد. از «ثبت سریع خوراکی دلخواه» استفاده کنید.")

    all_food_options = sorted(LABELS_FA.keys(), key=lambda label: to_persian(label))
    selected_foods = st.multiselect(
        "اقلام موجود در وعده",
        options=all_food_options,
        default=st.session_state.get("meal_selected_foods", []),
        format_func=to_persian,
        key=selected_widget_key,
        help=(
            "Food-101 مرجع اصلی غذاهای عمومی است. برای خانواده‌های واضح مثل پاستا/برگر/پیتزا، CLIP فقط همان خانواده را اصلاح می‌کند؛ "
            "در تصاویر مبهم، CLIP به‌عنوان متخصص غذاهای ایرانی وارد می‌شود. در بشقاب‌های چندجزئی هنوز اقلام دیگر را دستی اضافه کنید."
        ),
    )
    st.session_state["meal_selected_foods"] = selected_foods

    current_nutrition: dict[str, NutritionResult] = st.session_state.get(
        "meal_nutrition_by_label", {}
    )
    current_nutrition = {
        label: nutrition for label, nutrition in current_nutrition.items() if label in selected_foods
    }
    st.session_state["meal_nutrition_by_label"] = current_nutrition

    if selected_foods and st.button(
        "دریافت کالری اقلام",
        use_container_width=True,
        key=meal_builder_key("nutrition_button"),
    ):
        nutrition_by_label = dict(current_nutrition)
        failed_items: list[str] = []
        with st.spinner("در حال دریافت اطلاعات تغذیه‌ای..."):
            for label in selected_foods:
                if label in nutrition_by_label:
                    continue
                try:
                    nutrition_by_label[label] = lookup_calories(label)
                except NutritionLookupError:
                    failed_items.append(to_persian(label))

        st.session_state["meal_nutrition_by_label"] = nutrition_by_label
        if failed_items:
            st.warning("برای این موارد داده مناسب پیدا نشد: " + "، ".join(failed_items))
        if nutrition_by_label:
            st.success("اطلاعات کالری آماده شد.")

    nutrition_by_label: dict[str, NutritionResult] = st.session_state.get(
        "meal_nutrition_by_label", {}
    )
    ready_foods = [label for label in selected_foods if label in nutrition_by_label]

    if selected_foods and len(ready_foods) < len(selected_foods):
        st.info("برای ادامه، ابتدا اطلاعات کالری همه اقلام انتخاب‌شده را دریافت کنید.")

    if ready_foods:
        st.subheader("جزئیات وعده")
        with st.form(meal_builder_key("details_form")):
            top1, top2, top3 = st.columns(3)
            with top1:
                meal_type_fa = st.selectbox(
                    "نوع وعده",
                    options=[MEAL_TYPE_FA[key] for key in MEAL_ENTRY_TYPES],
                    index=meal_type_select_index(),
                )
            with top2:
                meal_date = st.date_input("تاریخ وعده", value=date.today())
            with top3:
                meal_time = st.time_input(
                    "زمان مصرف",
                    value=datetime.now().time().replace(second=0, microsecond=0),
                    step=timedelta(minutes=5),
                )
            notes = st.text_area(
                "یادداشت (اختیاری)",
                max_chars=500,
                placeholder="مثلاً بدون روغن یا بعد از تمرین",
            )

            form_values: dict[str, tuple[float, float, str]] = {}
            for index, label in enumerate(ready_foods, start=1):
                nutrition = nutrition_by_label[label]
                unit = measure_unit_for_label(label)
                st.markdown(f"**{index}. {to_persian(label)}**")
                if nutrition.source == "local_estimate":
                    st.caption(
                        f"برآورد داخلی: {nutrition.description} — این مقدار تقریبی و قابل ویرایش است."
                    )
                else:
                    st.caption(f"مرجع تغذیه‌ای: {nutrition.description}")
                if unit == "ml":
                    st.caption("🥤 این مورد نوشیدنی است؛ مقدار مصرف بر حسب میلی‌لیتر محاسبه می‌شود.")
                col_amount, col_kcal = st.columns(2)
                with col_amount:
                    amount = st.number_input(
                        f"{'حجم' if unit == 'ml' else 'وزن'} {to_persian(label)} ({short_unit_fa(unit)})",
                        min_value=1.0,
                        max_value=max_amount(unit),
                        value=default_amount(unit),
                        step=1.0,
                        format="%.0f",
                        key=meal_builder_key(f"amount_{label}"),
                        help=(
                            "حجم نوشیدنی را با دقت یک میلی‌لیتر؛ برای نمونه ۲۵۰ میلی‌لیتر وارد کنید."
                            if unit == "ml"
                            else "ورود وزن با دقت یک گرم؛ برای نمونه ۱۸۲ گرم."
                        ),
                    )
                with col_kcal:
                    reference_kcal = calories_for_display_unit(
                        label,
                        nutrition.calories_per_100g,
                        source=nutrition.source,
                    )
                    calories_per_100g = st.number_input(
                        f"کالری {to_persian(label)} در ۱۰۰ {short_unit_fa(unit)}",
                        min_value=0.0,
                        max_value=2000.0,
                        value=float(reference_kcal),
                        step=1.0,
                        key=meal_builder_key(f"kcal_{label}"),
                        help=(
                            "برای نوشیدنی‌ها مقدار مرجع به ازای ۱۰۰ میلی‌لیتر است؛ در صورت نیاز آن را اصلاح کنید."
                            if unit == "ml"
                            else "مقدار مرجع/برآورد به ازای ۱۰۰ گرم و قابل اصلاح است."
                        ),
                    )
                form_values[label] = (float(amount), float(calories_per_100g), unit)
                st.divider()

            submitted = st.form_submit_button(
                "ذخیره وعده",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            items: list[dict[str, Any]] = []
            for label, (amount, calories_per_100g, unit) in form_values.items():
                nutrition = nutrition_by_label[label]
                items.append(
                    {
                        "food_label": label,
                        "food_name_fa": to_persian(label),
                        "weight_grams": amount,
                        "measure_unit": unit,
                        "calories_per_100g": calories_per_100g,
                        "confidence": confidence_for_label(predictions, label),
                        "nutrition_description": nutrition.description,
                        "source": nutrition.source,
                    }
                )

            try:
                meal_id = create_meal(
                    user_id=int(user["id"]),
                    meal_type=MEAL_TYPE_BY_FA[meal_type_fa],
                    meal_date=meal_date,
                    meal_time=meal_time,
                    items=items,
                    image_hash=st.session_state.get("meal_image_hash"),
                    predicted_label=predictions[0].label if predictions else None,
                    notes=notes,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                total = sum(
                    item["weight_grams"] * item["calories_per_100g"] / 100.0
                    for item in items
                )
                clear_meal_builder(reset_form=True)
                st.session_state["flash_message"] = (
                    "success",
                    f"وعده شماره {meal_id} با مجموع تقریبی {total:,.0f} کیلوکالری ذخیره شد.",
                )
                st.rerun()


def render_history(user: dict[str, Any]) -> None:
    render_page_header("تاریخچه وعده‌ها", "مرور، فیلتر و مدیریت وعده‌های ثبت‌شده", "◷")
    default_start = date.today() - timedelta(days=6)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("از تاریخ", value=default_start, key="history_start")
    with col2:
        end_date = st.date_input("تا تاریخ", value=date.today(), key="history_end")

    if end_date < start_date:
        st.error("تاریخ پایان نباید قبل از تاریخ شروع باشد.")
        return
    if (end_date - start_date).days > 366:
        st.warning("برای عملکرد بهتر، بازه را حداکثر یک سال انتخاب کنید.")

    meals = get_meals_between(int(user["id"]), start_date, end_date)
    if not meals:
        st.info("در این بازه وعده‌ای ثبت نشده است.")
        return

    total = sum(float(meal["total_calories"]) for meal in meals)
    c1, c2 = st.columns(2)
    c1.metric("تعداد وعده‌ها", len(meals))
    c2.metric("مجموع کالری بازه", format_kcal(total))
    st.dataframe(meal_summary_table(meals), hide_index=True, use_container_width=True)

    st.subheader("جزئیات و مدیریت")
    for meal in meals:
        meal_id = int(meal["id"])
        meal_clock = f" {meal['meal_time']}" if meal.get("meal_time") else ""
        heading = (
            f"{meal['meal_date']}{meal_clock} | "
            f"{MEAL_TYPE_FA.get(meal['meal_type'], meal['meal_type'])} | "
            f"{float(meal['total_calories']):,.0f} kcal"
        )
        with st.expander(heading):
            items = get_meal_items(int(user["id"]), meal_id)
            st.dataframe(
                pd.DataFrame(
                    {
                        "غذا": [item["food_name_fa"] for item in items],
                        "مقدار مصرف": [
                            f"{float(item['weight_grams']):,.0f} {short_unit_fa(item.get('measure_unit', 'g'))}"
                            for item in items
                        ],
                        "مرجع کالری": [
                            f"{float(item['calories_per_100g']):,.1f} kcal / 100 {short_unit_fa(item.get('measure_unit', 'g'))}"
                            for item in items
                        ],
                        "کالری جزء": [
                            round(float(item["estimated_calories"]), 1) for item in items
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            if meal.get("notes"):
                st.caption(f"یادداشت: {meal['notes']}")
            confirm = st.checkbox(
                "حذف این وعده را تأیید می‌کنم",
                key=f"confirm_delete_{meal_id}",
            )
            if st.button(
                "حذف وعده",
                key=f"delete_meal_{meal_id}",
                disabled=not confirm,
            ):
                if delete_meal(int(user["id"]), meal_id):
                    st.session_state["flash_message"] = ("success", "وعده حذف شد.")
                    st.rerun()
                else:
                    st.error("حذف وعده ناموفق بود.")


def render_weekly_report(user: dict[str, Any]) -> None:
    render_page_header("گزارش هفتگی", "تحلیل مصرف از شنبه تا جمعه و مقایسه با هفته قبل", "▥")
    anchor = st.date_input("یک روز از هفته را انتخاب کنید", value=date.today())
    week_start, week_end = saturday_week_bounds(anchor)
    previous_start = week_start - timedelta(days=7)
    previous_end = week_start - timedelta(days=1)

    st.caption(
        f"هفته از شنبه تا جمعه · بازه گزارش: {week_start.isoformat()} تا {week_end.isoformat()}"
    )
    st.markdown(
        """
        <div class="hero-card">
          📅 مبنای گزارش هفتگی این برنامه مطابق تقویم رایج ایران، از <b>شنبه</b> آغاز و در <b>جمعه</b> پایان می‌یابد.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_rows = daily_totals(int(user["id"]), week_start, week_end)
    previous_rows = daily_totals(int(user["id"]), previous_start, previous_end)
    by_date = {row["meal_date"]: row for row in current_rows}

    chart_rows: list[dict[str, Any]] = []
    for offset in range(7):
        day = week_start + timedelta(days=offset)
        row = by_date.get(day.isoformat(), {})
        chart_rows.append(
            {
                "تاریخ": day.isoformat(),
                "روز": WEEKDAY_FA[day.weekday()],
                "کالری": float(row.get("total_calories", 0) or 0),
                "تعداد وعده": int(row.get("meal_count", 0) or 0),
            }
        )

    total = sum(row["کالری"] for row in chart_rows)
    logged = [row for row in chart_rows if row["تعداد وعده"] > 0]
    logged_days = len(logged)
    average_logged = total / logged_days if logged_days else 0.0
    goal = float(user["daily_calorie_goal"])
    days_at_or_below_goal = sum(1 for row in logged if row["کالری"] <= goal)
    previous_total = sum(float(row["total_calories"]) for row in previous_rows)

    delta_text: str | None = None
    if previous_total > 0:
        change = (total - previous_total) / previous_total * 100
        delta_text = f"{change:+.1f}٪ نسبت به هفته قبل"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("مجموع هفته", format_kcal(total), delta=delta_text)
    c2.metric("میانگین روزهای ثبت‌شده", format_kcal(average_logged))
    c3.metric("روزهای دارای ثبت", f"{logged_days} از ۷")
    c4.metric("روزهای زیر/برابر هدف", f"{days_at_or_below_goal} روز")

    if not logged:
        st.info("برای این هفته هنوز داده‌ای ثبت نشده است.")
        return

    highest = max(logged, key=lambda row: row["کالری"])
    lowest = min(logged, key=lambda row: row["کالری"])
    st.markdown(
        f"""
        <div class="soft-card">
          <b>بیشترین مصرف ثبت‌شده:</b> {highest['روز']}، {highest['کالری']:,.0f} kcal<br>
          <b>کمترین مصرف ثبت‌شده:</b> {lowest['روز']}، {lowest['کالری']:,.0f} kcal
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("روند روزانه")
    daily_df = pd.DataFrame(chart_rows)
    st.line_chart(daily_df.set_index("روز")[["کالری"]], use_container_width=True)
    st.dataframe(daily_df, hide_index=True, use_container_width=True)

    type_rows = meal_type_totals(int(user["id"]), week_start, week_end)
    if type_rows:
        st.subheader("سهم انواع وعده")
        type_df = pd.DataFrame(
            {
                "وعده": [MEAL_TYPE_FA.get(row["meal_type"], row["meal_type"]) for row in type_rows],
                "کالری": [round(float(row["total_calories"]), 1) for row in type_rows],
                "تعداد": [int(row["meal_count"]) for row in type_rows],
            }
        )
        st.bar_chart(type_df.set_index("وعده")[["کالری"]], use_container_width=True)
        st.dataframe(type_df, hide_index=True, use_container_width=True)

    foods = top_foods(int(user["id"]), week_start, week_end, limit=5)
    if foods:
        st.subheader("پرتکرارترین غذاهای هفته")
        st.dataframe(
            pd.DataFrame(
                {
                    "غذا": [row["food_name_fa"] for row in foods],
                    "دفعات مصرف": [int(row["times_consumed"]) for row in foods],
                    "مجموع کالری": [round(float(row["total_calories"]), 1) for row in foods],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_profile(user: dict[str, Any]) -> None:
    render_page_header("پروفایل و تنظیمات", "مدیریت حساب، رمز عبور و هدف کالری روزانه", "⚙")
    st.markdown(
        f"""
        <div class="soft-card">
          <b>نام کاربری فعلی:</b> {user['username']}<br>
          <b>ایمیل:</b> {user.get('email') or 'ثبت نشده'}<br>
          <b>تاریخ ساخت حساب:</b> {str(user['created_at'])[:10]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    account_tab, password_tab, goal_tab = st.tabs(
        ["اطلاعات حساب", "تغییر رمز عبور", "هدف روزانه"]
    )

    with account_tab:
        with st.form("account_profile_form"):
            username = st.text_input("نام کاربری", value=str(user["username"]))
            email = st.text_input("ایمیل *", value=str(user.get("email") or ""))
            account_submitted = st.form_submit_button(
                "ذخیره اطلاعات حساب", type="primary", use_container_width=True
            )
        if account_submitted:
            try:
                clean_username = validate_username(username)
                clean_email = validate_email(email)
                update_account_profile(
                    int(user["id"]),
                    username=clean_username,
                    email=clean_email,
                )
            except (AccountValidationError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["flash_message"] = (
                    "success",
                    "نام کاربری و اطلاعات حساب به‌روزرسانی شد.",
                )
                st.rerun()

    with password_tab:
        with st.form("change_password_form"):
            current_password = st.text_input("رمز عبور فعلی", type="password")
            new_password = st.text_input(
                "رمز عبور جدید (حداقل ۸ نویسه)", type="password"
            )
            confirmation = st.text_input("تکرار رمز عبور جدید", type="password")
            password_submitted = st.form_submit_button(
                "تغییر رمز عبور", type="primary", use_container_width=True
            )
        if password_submitted:
            credentials = get_user_credentials(int(user["id"]))
            if not credentials or not verify_password(
                current_password, credentials["password_hash"]
            ):
                st.error("رمز عبور فعلی درست نیست.")
            else:
                try:
                    validate_password(new_password)
                    if new_password != confirmation:
                        raise AccountValidationError("تکرار رمز عبور جدید یکسان نیست.")
                    if verify_password(new_password, credentials["password_hash"]):
                        raise AccountValidationError(
                            "رمز عبور جدید باید با رمز فعلی متفاوت باشد."
                        )
                    update_password_hash(int(user["id"]), hash_password(new_password))
                except (AccountValidationError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["flash_message"] = (
                        "success",
                        "رمز عبور با موفقیت تغییر کرد.",
                    )
                    st.rerun()

    with goal_tab:
        with st.form("daily_goal_form"):
            goal = st.number_input(
                "هدف کالری روزانه",
                min_value=500.0,
                max_value=10000.0,
                value=float(user["daily_calorie_goal"]),
                step=50.0,
            )
            goal_submitted = st.form_submit_button(
                "ذخیره هدف جدید", type="primary", use_container_width=True
            )
        if goal_submitted:
            try:
                update_daily_goal(int(user["id"]), float(goal))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["flash_message"] = (
                    "success",
                    "هدف روزانه به‌روزرسانی شد.",
                )
                st.rerun()

    st.info(
        "هدف کالری فقط برای مقایسه و نمایش گزارش استفاده می‌شود و توصیه پزشکی یا رژیم درمانی نیست."
    )


user = current_user()
if user is None:
    show_flash()
    render_authentication()
    st.stop()

page = render_top_navigation(user)
show_flash()

if page == "داشبورد امروز":
    render_dashboard(user)
elif page == "ثبت وعده جدید":
    render_meal_entry(user)
elif page == "تاریخچه":
    render_history(user)
elif page == "گزارش هفتگی":
    render_weekly_report(user)
elif page == "پروفایل و تنظیمات":
    render_profile(user)

st.divider()
st.caption(
    "مقادیر کالری تخمینی‌اند؛ دستور پخت، روغن، مواد اولیه و اندازه واقعی وعده می‌توانند نتیجه را تغییر دهند."
)
