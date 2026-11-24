"""Runtime readiness and setup instructions for the MAMMA model manager."""
from pathlib import Path
from html import escape
import json
from .theme import set_appearance_style
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QDialog,QHBoxLayout,QLabel,QPushButton,QTextBrowser,QVBoxLayout,QWidget
from .paths import ROOT


def runtime_panel(parent):
    panel=QWidget();box=QHBoxLayout(panel);box.setContentsMargins(0,0,0,0)
    label=QLabel();label.setWordWrap(True);label.setTextFormat(Qt.PlainText)
    try:
        from .mamma_runtime import configuration
        cfg=configuration()
        label.setText('MAMMA runtime: Ready' if cfg.get('version')==3 else 'MAMMA runtime: Configured separately')
        label.setToolTip('Python: '+cfg['python']+'\nEngine: '+cfg['engine']+'\nThis environment is isolated from the single-video models.')
    except (OSError,RuntimeError,ValueError) as error:
        label.setText('MAMMA runtime: Setup required');label.setToolTip(str(error));set_appearance_style(label, 'color:#ffb45c')
    button=QPushButton('Runtime Setup...')
    button.setToolTip('Show the separate MAMMA runtime installation steps, disk requirements and license conditions. Model weights are installed in the table below.')
    button.clicked.connect(lambda:setup_instructions(parent))
    box.addWidget(label,1);box.addWidget(button)
    return panel


def setup_instructions(parent):
    dialog=QDialog(parent);dialog.setWindowTitle('MAMMA Runtime Setup');dialog.resize(690,530)
    layout=QVBoxLayout(dialog);text=QTextBrowser();text.setOpenExternalLinks(True)
    text.document().setDefaultStyleSheet('a { color: #8ed2ff; text-decoration: underline; }')
    report={}
    try:report=json.loads((ROOT/'.setup/mamma-install-report.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):pass
    size=report.get('installed_bytes',0)
    measured=(f'{size/1024**3:.2f} GiB on this PC' if size else 'approximately 6–8 GiB')
    text.setHtml(f'''<h3>MAMMA separate runtime</h3>
<p>Installed size: <b>{measured}</b>, excluding model weights and developer tools.
Allow about <b>18 GiB free</b> during setup for downloads and compilation. Other model runtimes stay unchanged.</p>
<ol><li>Install <a href="https://visualstudio.microsoft.com/downloads/">Visual Studio 2022 Build Tools</a> with
<b>Desktop development with C++</b> and the Windows SDK.</li>
<li>Install <a href="https://developer.nvidia.com/cuda-12-8-0-download-archive">NVIDIA CUDA Toolkit 12.8</a>.
The GPU driver alone does not include the compiler.</li>
<li>Save your work, close RGB MoCap, then run <b>Setup_MAMMA.cmd</b> in the app folder.</li>
<li>Review the upstream license prompt. Setup downloads pinned Python/packages and official sources,
builds SAM 2 and SDF for this GPU, and tests the CUDA kernels before enabling the runtime.</li>
<li>Reopen the app. Select MAMMA and prepare the files in Model Manager.</li></ol>
<p><a href="https://github.com/cuevhv/mamma/blob/main/LICENSE">MAMMA</a> and
<a href="https://github.com/cuevhv/pytorch_sdf/blob/main/LICENSE">pytorch_sdf</a> have restricted, non-commercial licenses.
Their code and compiled modules are not redistributed by RGB MoCap. They are downloaded and built locally.
The compiler tools can add substantial disk usage beyond the runtime.</p>
<p>Registered MAMMA checkpoint, correspondence data and the three locked-head SMPL-X body files require
manual downloads. For the MAMMA checkpoint and surface landmarks, open the
<a href="https://mamma.is.tue.mpg.de/">official MAMMA website</a>, click <b>Data</b>,
register if needed, review and accept the provider terms, then sign in to download the files.
Return to Model Manager and use <b>Import</b> for each matching row.
The locked-head body files come from the separate SMPL-X website.
The table provides instructions and SHA-256 checked imports. Public supporting weights can be downloaded there.</p>
<p>Setup logs: <code>.setup/logs/mamma-*.log</code>.<br>Details: <code>docs/MAMMA.md</code>.</p>''')
    layout.addWidget(text);buttons=QHBoxLayout()
    copy=QPushButton('Copy Setup Command');copy.setToolTip('Copy a command to run after closing RGB MoCap.')
    copy.clicked.connect(lambda:QApplication.clipboard().setText('cd /d "'+str(ROOT)+'"\r\nSetup_MAMMA.cmd'))
    close=QPushButton('Close');close.clicked.connect(dialog.accept)
    buttons.addWidget(copy);buttons.addStretch();buttons.addWidget(close);layout.addLayout(buttons)
    dialog.exec();dialog.deleteLater()
