"""Theme tokens and QSS stylesheet for the SensiExtractor GUI.

Light theme only. The dark variant and OS-follow logic were removed in
batch 3 — the lab wanted a single calm light surface (#F4F6F8) rather
than a system-tracking palette.
"""

from pathlib import Path


ACCENT_COLOR = "#00B2D0"      # SENSI cyan — matches assets/icon.svg
ACCENT_HOVER = "#0093AB"
ACCENT_DISABLED = "#B0DCE5"
ACCENT_SOFT = "#E0F4F8"        # tinted background for selected sidebar item

BG_PAGE = "#F4F6F8"            # main content area background (calm, not bright)
BG_PANEL = "#FFFFFF"           # card / panel background
BG_SIDEBAR = "#EDEFF2"         # sidebar background
BG_INPUT = "#FFFFFF"

BORDER_DEFAULT = "#D8DCE0"
BORDER_STRONG = "#9CA3AF"
DIVIDER_COLOR = "#C9CED4"      # section dividers — needs to be visible

TEXT_PRIMARY = "#1F2328"
TEXT_SECONDARY = "#5B6370"
TEXT_MUTED = "#8A8F98"
TEXT_DISABLED = "#B4B9BF"      # greyed controls (left-out section bodies)

ERROR_RED = "#B0312A"
AMBER = "#C77A00"

GRID_UNIT = 8
CARD_RADIUS = 8
BUTTON_RADIUS = 6
INPUT_RADIUS = 4

FONT_BODY = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12)
FONT_MONO = ("JetBrains Mono", 10)


# Checkbox/radio indicator images. PySide6 6.11 ignores data: URIs in QSS
# `image:` rules, so we point at real SVG files in assets/. Path is computed
# absolutely so Qt can always locate the file regardless of CWD.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
# Qt QSS url() uses forward slashes even on Windows.
_CHECK_URL = "url('" + (_ASSETS_DIR / "check.svg").as_posix() + "')"
_DASH_URL = "url('" + (_ASSETS_DIR / "dash.svg").as_posix() + "')"
_DOT_URL = "url('" + (_ASSETS_DIR / "dot.svg").as_posix() + "')"


_QSS = f"""
QMainWindow {{ background: {BG_PAGE}; color: {TEXT_PRIMARY}; }}
QStackedWidget {{ background: {BG_PAGE}; }}
QScrollArea {{ background: {BG_PAGE}; border: none; }}

QFrame#Sidebar {{
    background: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_DEFAULT};
}}
QFrame#TopBar {{
    background: {BG_PANEL};
    border-bottom: 1px solid {BORDER_DEFAULT};
}}
QFrame#StatusStrip {{
    background: {BG_PANEL};
    border-top: 1px solid {BORDER_DEFAULT};
}}
QFrame#Card {{
    background: {BG_PANEL};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {CARD_RADIUS}px;
}}
QFrame#SectionDivider {{
    background: {DIVIDER_COLOR};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QListWidget {{
    background: {BG_SIDEBAR};
    border: none;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 6px;
    min-height: 20px;
    color: {TEXT_PRIMARY};
}}
QListWidget::item:selected {{
    background: {ACCENT_SOFT};
    color: {ACCENT_COLOR};
    font-weight: 600;
}}
QListWidget::item:hover:!selected {{
    background: rgba(0, 178, 208, 0.08);
}}

QPushButton {{
    background: {BG_PANEL};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {BUTTON_RADIUS}px;
    padding: 6px 12px;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover {{ background: #F2F4F7; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}

QPushButton#Primary {{
    background: {ACCENT_COLOR};
    color: white;
    padding: 8px 16px;
    border-radius: {BUTTON_RADIUS}px;
    border: none;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#Primary:disabled {{ background: {ACCENT_DISABLED}; }}

QPushButton#InfoIcon {{
    background: transparent;
    border: none;
    padding: 0;
    border-radius: 11px;
}}
QPushButton#InfoIcon:hover {{ background: {ACCENT_SOFT}; }}

QLabel {{ color: {TEXT_PRIMARY}; }}
QLabel#SectionHeader {{
    font-size: 14px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding-bottom: 2px;
}}
QLabel#Subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}
/* Greyed (disabled) text — the generic QLabel colour rule above would
   otherwise override Qt's disabled palette, leaving labels looking live. */
QLabel:disabled, QLabel#Subtitle:disabled, QLabel#GroupLabel:disabled,
QLabel#GroupLinkSep:disabled {{ color: {TEXT_DISABLED}; }}

/* Section card (ClusterCard): title → switch → tag header. 4 px left edge
   cyan when included, grey (+ grey header) when left out. */
QFrame#SectionCardHeader {{
    background: transparent;
    border: none;
    border-bottom: 1px solid #ECEFF1;
    border-left: 4px solid {ACCENT_COLOR};
    border-top-left-radius: {CARD_RADIUS - 1}px;
}}
QFrame#SectionCardHeader[included="false"] {{
    background: #F4F5F6;
    border-left: 4px solid #B9C0C5;
}}
QLabel#SectionCardTitle {{
    font-size: 14px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
QLabel#SectionTag {{
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9px;
    background: #E0F6FA;
    color: #00737F;
}}
QLabel#SectionTag[state="off"] {{
    background: #E6E8EA;
    color: #5B6368;
}}
QLabel#GroupLabel {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
}}
QLabel#GroupLinkSep {{ color: {TEXT_MUTED}; font-size: 11px; }}
QPushButton#GroupLink {{
    background: transparent;
    border: none;
    padding: 0 2px;
    color: #00838F;
    font-size: 11px;
}}
QPushButton#GroupLink:hover {{ text-decoration: underline; background: transparent; }}
QPushButton#GroupLink:disabled {{ color: {TEXT_DISABLED}; }}
QFrame#SharedSettings {{
    background: {BG_PAGE};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {INPUT_RADIUS}px;
}}
QLabel#SharedNote {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
}}

/* QComboBox — needs both the button face AND the popup view styled.
   Without QAbstractItemView rules the popup inherits Qt's default dark
   palette on Windows and the text becomes invisible. */
QComboBox {{
    padding: 6px 8px;
    background: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {INPUT_RADIUS}px;
}}
QComboBox:focus {{ border-color: {ACCENT_COLOR}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_DEFAULT};
    selection-background-color: {ACCENT_COLOR};
    selection-color: white;
    outline: none;
}}

QLineEdit, QPlainTextEdit, QTextEdit {{
    padding: 6px 8px;
    background: {BG_INPUT};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {INPUT_RADIUS}px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_COLOR};
    selection-color: white;
}}
QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT_COLOR};
}}
QLineEdit[error="true"] {{ border-color: {ERROR_RED}; }}

/* Checkboxes — explicit indicator styling required, otherwise Qt suppresses
   the native indicator once any QCheckBox property is set in QSS. */
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 10px;
    padding: 3px 0;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {BORDER_STRONG};
    border-radius: 3px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT_COLOR}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT_COLOR};
    border-color: {ACCENT_COLOR};
    image: {_CHECK_URL};
}}
QCheckBox::indicator:checked:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QCheckBox::indicator:indeterminate {{
    background: {ACCENT_COLOR};
    border-color: {ACCENT_COLOR};
    image: {_DASH_URL};
}}
QCheckBox::indicator:disabled {{
    background: #E5E8EB;
    border-color: {BORDER_DEFAULT};
}}
QCheckBox:disabled {{ color: {TEXT_MUTED}; }}

QRadioButton {{
    color: {TEXT_PRIMARY};
    spacing: 10px;
    padding: 3px 0;
    background: transparent;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 9px;
    border: 1.5px solid {BORDER_STRONG};
    background: {BG_INPUT};
}}
QRadioButton::indicator:hover {{ border-color: {ACCENT_COLOR}; }}
QRadioButton::indicator:checked {{
    border: 1.5px solid {ACCENT_COLOR};
    background: {BG_INPUT};
    image: {_DOT_URL};
}}

QTableView {{
    background: {BG_PANEL};
    border: 1px solid {BORDER_DEFAULT};
    gridline-color: {BORDER_DEFAULT};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background: #EAECEF;
    color: {TEXT_PRIMARY};
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {BORDER_DEFAULT};
    font-weight: 600;
}}

QProgressBar {{
    background: #E5E8EB;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: {TEXT_PRIMARY};
}}
QProgressBar::chunk {{
    background: {ACCENT_COLOR};
    border-radius: 4px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_DEFAULT};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {BORDER_STRONG}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_DEFAULT};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {BORDER_STRONG}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def qss_for_appearance(_mode: str = "light") -> str:
    """Return the light-theme stylesheet.

    The `_mode` argument is accepted for backward compat with callers that
    still pass 'light' / 'dark' / 'system'. Dark mode was removed in batch 3;
    every call returns the light stylesheet.
    """
    return _QSS


def detect_system_appearance() -> str:
    """Kept for backward compat; always returns 'light'."""
    return "light"
