from pathlib import Path

APP_TEXT = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")


def test_native_sidebar_is_not_used_for_navigation():
    assert "with st.sidebar:" not in APP_TEXT
    assert "def render_top_navigation" in APP_TEXT
    assert 'section[data-testid="stSidebar"]' in APP_TEXT
    assert "display: none !important" in APP_TEXT


def test_manual_blue_black_dashboard_ui_is_present():
    assert "--blue-500: #2f80ff" in APP_TEXT
    assert "--ink-950: #030711" in APP_TEXT
    assert 'class="metric-grid"' in APP_TEXT
    assert 'class="progress-track"' in APP_TEXT


def test_mobile_hamburger_navigation_is_present():
    assert 'key="mobile_menu_toggle"' in APP_TEXT
    assert '"☰ منو"' in APP_TEXT
    assert 'mobile_menu_open' in APP_TEXT
    assert 'on_change=close_mobile_menu' in APP_TEXT


def test_dark_light_toggle_is_present_and_session_based():
    assert 'key="theme_toggle"' in APP_TEXT
    assert 'ui_theme' in APP_TEXT
    assert 'toggle_ui_theme' in APP_TEXT
    assert 'current_theme == "light"' in APP_TEXT
    assert '"☀️" if current_theme == "light" else "🌙"' in APP_TEXT
    assert '☀️ روشن' not in APP_TEXT
    assert '🌙 تاریک' not in APP_TEXT


def test_light_theme_overrides_are_present():
    assert '--ink-950: #eef5ff' in APP_TEXT
    assert 'background: #eef5ff !important' in APP_TEXT
    assert 'button[kind="secondary"]' in APP_TEXT
    assert '.st-key-theme_toggle button' in APP_TEXT
    assert 'key="top_navbar"' in APP_TEXT
    assert '[data-testid="stTextInput"] input' in APP_TEXT
    assert '[data-testid="stNumberInput"] button' in APP_TEXT
    assert 'mobile_display = "block" if menu_open else "none"' in APP_TEXT
