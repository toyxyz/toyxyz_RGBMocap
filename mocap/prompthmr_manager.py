"""PromptHMR optional-runtime readiness and concise installation guidance."""
from PySide6.QtWidgets import QWidget,QHBoxLayout,QLabel,QPushButton,QDialog,QVBoxLayout,QTextBrowser,QApplication
from .paths import ROOT
import json


def runtime_panel(parent):
    panel=QWidget();layout=QHBoxLayout(panel);layout.setContentsMargins(0,0,0,0);label=QLabel();label.setWordWrap(True)
    try:
        from .prompthmr_runtime import configuration
        cfg=configuration();label.setText('PromptHMR runtime: Ready');label.setToolTip('Isolated Python: '+cfg['python'])
        try:
            from .prompthmr_camera_runtime import configuration as camera_configuration
            camera_configuration();label.setText(label.text()+'\nMoving camera support: Ready')
        except (OSError,ValueError,RuntimeError):label.setText(label.text()+'\nMoving camera support: Optional setup')
    except (OSError,ValueError,RuntimeError) as error:
        label.setText('PromptHMR runtime: Setup required');label.setToolTip(str(error));label.setStyleSheet('color:#ffb45c')
    button=QPushButton('Runtime Setup...');button.setToolTip('Show separate runtime setup, disk space and license requirements. Weights are prepared below.')
    button.clicked.connect(lambda:setup_instructions(parent));layout.addWidget(label,1);layout.addWidget(button);return panel


def setup_instructions(parent):
    dialog=QDialog(parent);dialog.setWindowTitle('PromptHMR Runtime Setup');dialog.resize(720,530);layout=QVBoxLayout(dialog)
    browser=QTextBrowser();browser.setOpenExternalLinks(True);browser.document().setDefaultStyleSheet('a { color:#8ed2ff; text-decoration:underline; }')
    size='approximately 6–8 GiB'
    try:
        report=json.loads((ROOT/'.setup/prompthmr-install-report.json').read_text(encoding='utf8'));size=f"{report['installed_bytes']/2**30:.2f} GiB on this PC"
    except (OSError,ValueError,KeyError):pass
    browser.setHtml(f'''<h3>PromptHMR separate Windows runtime</h3><p>Installed runtime: <b>{size}</b>, excluding weights and compiler tools. Allow about <b>18 GiB free</b> during installation. Other model environments stay unchanged.</p>
<ol><li>Run install.cmd for the base app.</li><li>Install <a href="https://visualstudio.microsoft.com/downloads/">Visual Studio 2022 Build Tools</a> with Desktop development with C++ and Windows SDK, and <a href="https://developer.nvidia.com/cuda-12-8-0-download-archive">CUDA Toolkit 12.8</a>.</li><li>Close RGB MoCap and run <b>Setup_PromptHMR.cmd</b>. Review the official license prompt. Setup downloads pinned sources/packages, builds SAM2 for this GPU and verifies CUDA operations. Choose Y at the optional camera prompt to also build DroidCalib/LieTorch (AGPL/BSD; review their licenses). Camera support reuses this Python/Torch, needs about 6 GiB extra build headroom and 1.55 GiB of additional camera weights. Fixed Camera does not require these extras.</li><li>Reopen the app. Use Model Manager to download or import every required file. Existing standard SMPL-X Neutral and SMPL Neutral body files are reused; MAMMA's locked-head body is different.</li></ol>
<p><a href="https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE">PromptHMR is restricted to non-commercial scientific research</a>. Its source, weights and registered body data are not redistributed by RGB MoCap. Other components keep their own licenses.</p><p>Python 3.11 / Torch 2.7.1 CUDA 12.8 / timm 0.9.0. Torch's version overlaps with other profiles, but Python ABI and package versions differ. Installed packages remain isolated; verified identical download archives can be reused. MAMMA/SDF is not required.</p><p>Current support: one video with a fixed or optionally tracked moving camera, automatic tracking of up to eight people, shared world placement, optional interaction prompt, correction and FBX. Optional Moving Camera adds DroidCalib + Metric3D tracking; no animated finger inference. Details: docs/models/PROMPTHMR.md. Logs: .setup/logs/prompthmr-*.log.</p>''')
    layout.addWidget(browser);buttons=QHBoxLayout();copy=QPushButton('Copy Setup Command');copy.clicked.connect(lambda:QApplication.clipboard().setText('cd /d "'+str(ROOT)+'"\r\nSetup_PromptHMR.cmd'));close=QPushButton('Close');close.clicked.connect(dialog.accept)
    buttons.addWidget(copy);buttons.addStretch();buttons.addWidget(close);layout.addLayout(buttons);dialog.exec();dialog.deleteLater()
