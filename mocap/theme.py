"""Compact neutral workspace theme."""
from PySide6.QtCore import QLocale
from PySide6.QtGui import QColor, QFont, QPalette, QFontDatabase
from .paths import ROOT

STYLE = """
QWidget { color:#d5d5d5; font-size:12px; }
QMainWindow,QDialog { background:#252525; }
QLabel#brand { font-size:13px; font-weight:600; color:#ededed; padding:0 8px; }
QLabel#muted { color:#a0a0a0; }
QLabel#status { color:#b8b8b8; padding:3px 6px; }
QLabel#taskInfo { color:#b9d9ff; padding:3px 6px; }
QFrame#editor { background:#303030; border:1px solid #191919; }
QWidget#editorHeader { background:#292929; border-bottom:1px solid #191919; }
QGroupBox { border:1px solid #202020; border-radius:2px; margin-top:17px; padding:9px 7px 7px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 3px; color:#c5c5c5; }
QPushButton { background:#454545; border:1px solid #242424; border-radius:3px; padding:5px 9px; }
QPushButton:hover { background:#555555; }
QPushButton:pressed { background:#333333; }
QPushButton:disabled { color:#747474; background:#353535; }
QPushButton#primary { background:#477bb7; border-color:#37689d; color:white; font-weight:600; }
QPushButton#primary:hover { background:#548dce; }
QPushButton#transport { padding:3px; min-width:25px; }
QComboBox,QAbstractSpinBox { background:#454545; border:1px solid #242424; border-radius:3px; padding:3px 5px; min-height:20px; }
QComboBox:hover,QAbstractSpinBox:hover { border-color:#737373; }
QComboBox QAbstractItemView { background:#333333; selection-background-color:#477bb7; }
QCheckBox { spacing:6px; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #949494; border-radius:2px; background:#303030; }
QCheckBox::indicator:hover { border-color:#d8e9ff; background:#404c59; }
QCheckBox::indicator:checked { border-color:#c7ddf6; background:#477bb7; image:url("CHECK_ICON"); }
QCheckBox::indicator:disabled { border-color:#666666; background:#353535; }
QListWidget#peopleList::item { min-height:24px; }
QListWidget#peopleList::indicator { width:16px; height:16px; border:1px solid #949494; border-radius:2px; background:#303030; }
QListWidget#peopleList::indicator:hover { border-color:#d8e9ff; background:#404c59; }
QListWidget#peopleList::indicator:checked { border-color:#c7ddf6; background:#477bb7; image:url("CHECK_ICON"); }
QListWidget#peopleList::indicator:disabled { border-color:#666666; background:#353535; }
QProgressBar { background:#242424; border:0; height:3px; }
QProgressBar::chunk { background:#658fbd; }
QTextEdit,QTableWidget { background:#242424; border:1px solid #181818; selection-background-color:#477bb7; }
QTextEdit { font-family:Consolas; font-size:11px; }
QHeaderView::section { background:#333333; border:0; padding:6px; }
QMenuBar { background:#222222; border-bottom:1px solid #171717; }
QMenuBar::item { padding:4px 8px; }
QMenuBar::item:selected,QMenu::item:selected { background:#477bb7; }
QMenu { background:#303030; border:1px solid #171717; padding:4px; }
QMenu::item { padding:5px 22px; }
QSplitter::handle { background:#202020; }
QScrollArea { background:#303030; border:0; }
QWidget#modelSidebar,QStackedWidget#modelPages,QWidget[motion_tool],
QWidget#modelSettingsViewport,QFrame#modelActions { background:#303030; }
QWidget#modelSidebar QGroupBox { background:#303030; }
QScrollBar:vertical { width:9px; background:#292929; }
QScrollBar::handle:vertical { background:#555555; border-radius:3px; min-height:25px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QToolTip { background:#202020; color:#eeeeee; border:1px solid #555555; padding:5px; }
"""


def configure_application(app):
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app.setStyle("Fusion")
    # Bundled OFL font also works in the offscreen renderer without system fonts.
    font_id = QFontDatabase.addApplicationFont(str(ROOT / "assets/fonts/NotoSansKR.ttf"))
    family = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    app.setFont(QFont(family[0] if family else "Segoe UI", 9))
    palette = QPalette()
    for role, color in ((QPalette.Window, "#303030"), (QPalette.WindowText, "#d5d5d5"),
                        (QPalette.Base, "#242424"), (QPalette.AlternateBase, "#333333"),
                        (QPalette.Text, "#d5d5d5"), (QPalette.Button, "#454545"),
                        (QPalette.ButtonText, "#d5d5d5"), (QPalette.Highlight, "#477bb7"),
                        (QPalette.HighlightedText, "#ffffff")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE.replace('CHECK_ICON',(ROOT/'assets/icons/check.svg').as_posix()))
