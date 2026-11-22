"""Application appearance and installation-local preferences."""
import re

from PySide6.QtCore import QLocale
from PySide6.QtGui import QColor, QFont, QPalette, QFontDatabase
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QMessageBox, QVBoxLayout, QWidget)
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
QTabWidget#peopleTabs::pane { background:#303030; border:0; }
QTabWidget#peopleTabs QTabBar::tab, QTabBar#modelManagerTabs::tab { background:#292929; border:1px solid #191919; padding:5px 11px; }
QTabWidget#peopleTabs QTabBar::tab:!selected:hover, QTabBar#modelManagerTabs::tab:!selected:hover { background:#454545; }
QTabWidget#peopleTabs QTabBar::tab:selected, QTabBar#modelManagerTabs::tab:selected { background:#477bb7; border-color:#37689d; color:white; }
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
QListWidget#peopleList::item,QListWidget#streamPeopleList::item { min-height:24px; }
QListWidget#peopleList::indicator,QListWidget#streamPeopleList::indicator { width:16px; height:16px; border:1px solid #949494; border-radius:2px; background:#303030; }
QListWidget#peopleList::indicator:hover,QListWidget#streamPeopleList::indicator:hover { border-color:#d8e9ff; background:#404c59; }
QListWidget#peopleList::indicator:checked,QListWidget#streamPeopleList::indicator:checked { border-color:#c7ddf6; background:#477bb7; image:url("CHECK_ICON"); }
QListWidget#peopleList::indicator:disabled,QListWidget#streamPeopleList::indicator:disabled { border-color:#666666; background:#353535; }
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


# Keep the workspace geometry shared across themes.
LIGHT_COLORS = {
    '#d5d5d5': '#222b35', '#252525': '#f5f6f8', '#ededed': '#17202a',
    '#a0a0a0': '#556271', '#b8b8b8': '#43515e', '#b9d9ff': '#245786',
    '#303030': '#f1f3f6', '#191919': '#c2cad3', '#292929': '#e9edf2',
    '#454545': '#e3e8ee', '#477bb7': '#2866a3', '#37689d': '#205588',
    '#202020': '#d1d8e0', '#c5c5c5': '#364452', '#242424': '#ffffff',
    '#555555': '#c4d2e1', '#333333': '#e5ebf2', '#747474': '#626e79',
    '#353535': '#edf0f3', '#548dce': '#3579b9', '#737373': '#627789',
    '#949494': '#8595a5', '#d8e9ff': '#47729b', '#404c59': '#e0ebf7',
    '#c7ddf6': '#174674', '#666666': '#a9b2bc', '#658fbd': '#397eb9',
    '#181818': '#c0c9d3', '#222222': '#ebeff3', '#171717': '#bfc8d2',
    '#eeeeee': '#24313d', '#8ed2ff': '#125594', '#ffb45c': '#8a4809',
    '#f2b899': '#8c351c', '#303f50': '#dce9f7',
    '#999999': '#52616e', '#b3b3b3': '#42505c', '#aaaaaa': '#475663',
    '#b1b9c1': '#314c64', '#aebeb4': '#345846', '#a6bed5': '#426e96',
    '#74a6e0': '#2866a3', '#777777': '#5b6874', '#d9e8fa': '#284e74',
    '#383838': '#e6eaf0', '#515151': '#afbcca', '#bebebe': '#495768',
    '#bcbcbc': '#465565', '#afbdd2': '#445b72', '#dddddd': '#354a5f',
    '#dedede': '#24384c', '#a8a8a8': '#4c5b6a', '#e0b06c': '#824508',
    '#ffcc82': '#864404', '#e8c38d': '#895018',
}
LIGHT_STYLE = """
QPushButton,QComboBox,QAbstractSpinBox { border-color:#aab6c2; }
QPushButton:hover { background:#d5e2ef; }
QPushButton:pressed { background:#c3d4e5; }
QPushButton:disabled { border-color:#cbd2da; }
QMenuBar::item:selected,QMenu::item:selected { color:#ffffff; }
QComboBox QAbstractItemView,QTextEdit,QTableWidget { selection-color:#ffffff; }
"""

DEFAULT_THEME = 'dark'
DEFAULT_SCALE = 100
UI_SCALES = (75, 90, 100, 110, 125, 150, 175, 200)
THEMES = (('Dark', 'dark'), ('Light', 'light'))


def load_appearance():
    from .resource_safety import path, read_settings

    try:
        settings = read_settings(path())
    except (OSError, ValueError):
        return DEFAULT_THEME, DEFAULT_SCALE
    theme = settings.get('ui_theme')
    scale = settings.get('ui_scale')
    return (
        theme if theme in ('dark', 'light') else DEFAULT_THEME,
        scale if type(scale) is int and scale in UI_SCALES else DEFAULT_SCALE,
    )


def current_appearance():
    app = QApplication.instance()
    if app is None:
        return DEFAULT_THEME, DEFAULT_SCALE
    return app.property('uiTheme') or DEFAULT_THEME, app.property('uiScale') or DEFAULT_SCALE


def ui_px(pixels):
    return round(pixels * current_appearance()[1] / 100)


def is_light_theme():
    return current_appearance()[0] == 'light'


def theme_color(color, light=None):
    return (light or LIGHT_COLORS.get(color.lower(), color)) if is_light_theme() else color


def themed_colors(text):
    if not is_light_theme():
        return text
    return re.sub(r'#[0-9a-fA-F]{6}\b', lambda match: LIGHT_COLORS.get(match[0].lower(), match[0]), text)


def appearance_style(style):
    scale = current_appearance()[1]
    style = themed_colors(style)
    if scale == DEFAULT_SCALE:
        return style
    return re.sub(r'([1-9]\d*)px\b', lambda match: f'{max(1, round(int(match[1]) * scale / 100))}px', style)


def set_appearance_style(widget, style):
    widget.setProperty('baseAppearanceStyle', style)
    widget.setStyleSheet(appearance_style(style))


class AppearanceDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.theme = QComboBox()
        for label, value in THEMES:
            self.theme.addItem(label, value)
        self.scale = QComboBox()
        for value in UI_SCALES:
            self.scale.addItem(f'{value}%', value)
        self.scale.setToolTip('Scales text, buttons and panels relative to the display scale.')

        form = QFormLayout()
        form.addRow('Theme', self.theme)
        form.addRow('UI Scale', self.scale)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.default_button = buttons.addButton('Default', QDialogButtonBox.ButtonRole.ResetRole)
        self.default_button.clicked.connect(lambda: self.set_choices(DEFAULT_THEME, DEFAULT_SCALE))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.set_choices(*load_appearance())

    def set_choices(self, theme, scale):
        self.theme.setCurrentIndex(self.theme.findData(theme))
        self.scale.setCurrentIndex(self.scale.findData(scale))

    def accept(self):
        from .resource_safety import update

        values = (self.theme.currentData(), self.scale.currentData())
        if values != load_appearance():
            try:
                update({'ui_theme': values[0], 'ui_scale': values[1]})
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, 'Settings', f'Could not save settings.\n{error}')
                return
        super().accept()


def configure_application(app, theme=DEFAULT_THEME, scale=DEFAULT_SCALE):
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app.setStyle("Fusion")
    # Bundled OFL font also works in the offscreen renderer without system fonts.
    font_id = QFontDatabase.addApplicationFont(str(ROOT / "assets/fonts/NotoSansKR.ttf"))
    family = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    app.setFont(QFont(family[0] if family else "Segoe UI", 9))
    apply_appearance(app, theme, scale)


def apply_appearance(app, theme, scale):
    app.setProperty('uiTheme', theme)
    app.setProperty('uiScale', scale)
    font = app.font()
    font.setPointSizeF(9 * scale / 100)
    app.setFont(font)
    palette = app.style().standardPalette()
    for role, color in ((QPalette.Window, "#303030"), (QPalette.WindowText, "#d5d5d5"),
                        (QPalette.Base, "#242424"), (QPalette.AlternateBase, "#333333"),
                        (QPalette.Text, "#d5d5d5"), (QPalette.Button, "#454545"),
                        (QPalette.ButtonText, "#d5d5d5"), (QPalette.Highlight, "#477bb7"),
                        (QPalette.HighlightedText, "#ffffff")):
        palette.setColor(role, QColor(theme_color(color)))
    if theme == "light":
        palette.setColor(QPalette.Link, QColor(theme_color("#8ed2ff")))
    app.setPalette(palette)
    style = appearance_style(STYLE) + (LIGHT_STYLE if theme == "light" else "")
    app.setStyleSheet(style.replace('CHECK_ICON',(ROOT/'assets/icons/check.svg').as_posix()))
    for widget in app.allWidgets():
        style = widget.property('baseAppearanceStyle')
        if style:
            widget.setStyleSheet(appearance_style(style))
        QWidget.update(widget)
