"""Compact model comparison with recorded input and measurement conditions."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QTextOption
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout
from .motion_tools import MOTION_TOOLS

# exmr-eval-2026-10-22/summary-body.json, clip_01 (210 frames at 30 fps).
# Historical processing-only runs; FlowHMR predates Fast SAM.
# MAMMA: separate 413-frame/three-view manual run (SAM 2.1, VGGT, two people).
# Time includes correction/preview; memory is measured across four inference workers.
BENCHMARKS = {
    'pear': ('5s live video<br>1 person','21.43 ms/frame median<br>23.80 ms p95','1.50','—'),
    'gvhmr': ('7s<br>1 video', '2m 01s', '5.08', '12.47'),
    'flowhmr': ('7s<br>1 video', '2m 49s', '14.89', '22.19'),
    'exmr': ('7s<br>1 video', '1m 08s', '11.32', '19.31'),
    'mamma': ('13.8s<br>3 videos<br>2 people', '11m 52s', '6.47', '9.11'),
    'prompthmr': ('14.1s<br>1 video<br>2 people', '3m 56s', '4.61', '8.74'),
}
PROJECT_URLS = {
    'pear':'https://github.com/Pixel-Talk/PEAR',
    'prompthmr': 'https://github.com/yufu-wang/PromptHMR',
    'mamma': 'https://mamma.is.tue.mpg.de/',
    'gvhmr': 'https://github.com/zju3dv/GVHMR',
    'flowhmr': 'https://github.com/flowhmr/flowhmr',
    'exmr': 'https://github.com/twehrbein/ExMR',
}
DESCRIPTIONS = {
    'pear':'Experimental support. Tracking and pose estimates may be unstable. Real-time tracking of webcam or video input, with a single-person body and finger bone preview from one or multiple fixed-camera views. Select synchronized videos with Open Video, or assign distinct webcam devices using the camera-count slots and thumbnails. Start Live Tracking attempts Camera Align first, including one camera; VGGT and GeoCalib are needed for alignment. A single view uses estimated body/depth scale; multiple views use triangulation. If alignment fails or its support/models are missing, tracking continues from the selected camera with a visible warning, without camera fusion or aligned foot lock. Use Camera Align to retry. Fixed bone lengths are calibrated once; Reset Bones recalibrates from the displayed pose, then aligns cameras again. World Position is on by default; off fixes pelvis translation while preserving rotation. Body mesh is not computed per live frame. Select a webcam in the table below the video for image preview without loading PEAR; click Start Live Tracking to attempt alignment and begin tracking. Capture stays connected during Start and Camera Align; opened videos can play while models load or align. Choose None for video. Stop releases models and the webcam. Uses YOLOv8s person detection; detector boxes can affect the resulting pose. Rotation and One Euro Filter apply to both sources. One Euro Filter has independent Bone Rotation and World Position checkboxes, both on by default, with shared Filter Strength. The position-filter checkbox is separate from the World Position movement switch. Filtering reduces tremors with some motion-dependent lag and no extra inference. Seeks start from the selected frame; filtering continues during playback. After successful alignment, default-on lightweight Foot Lock stabilizes low, slow feet with smooth causal transitions, fixed lengths and preserved ankle/toe directions. Reach limits and incorrect ground estimates can leave residual sliding. No Find person, recording, offline Motion Correction, retargeting or motion export in this live mode. Single-view pelvis placement assumes a fixed camera. Camera Align estimates background gravity, camera positions, scale and floor; these are not measured calibration and feet can still slide. Changed inputs and Reset Bones trigger realignment. After moving physical cameras, click Camera Align again. Webcam arrival-time pairing is approximate, not hardware sync. Support files are included in install.cmd and update.cmd, reusing the shared runtime without upgrading its packages or requiring a CUDA compiler. Prepare the checkpoint and registered SMPL-X 2020/FLAME geometry bundle in Model Manager. See docs/models/PEAR.md.',
    'prompthmr': 'Reconstructs multiple people together from one video. In our subjective tests, it could give more accurate relative placement and interactions, but overall motion quality was lower than ExMR. Contacts, depth and intersections can still be wrong. Fixed Camera is the default; optional Moving Camera uses masked DroidCalib + Metric3D, with a separate camera build and two additional models. Estimated tracking/scale can drift. People defaults to 2; 0 detects automatically, up to 8. Find person is not needed. Interaction Prompt defaults on; off follows the official default video path. Body motion uses SMPL-X. No finger tracking; untracked fingers use a fixed neutral pose. Supports the usual optional foot/hand contact correction, retargeting and FBX/BVH/Blender export. Requires Setup_PromptHMR.cmd and separate weights. Non-commercial scientific research license. See docs/models/PROMPTHMR.md.',
    'mamma': 'Multi-view reconstruction of one or several people from two or more synchronized fixed-camera videos of the same performance. No fixed camera-count maximum; practical capacity depends on available RAM and VRAM. Detects people automatically; Find person is not needed. Uses SAM 2.1 masks and FP32 landmarks. Automatic cameras combine VGGT-Omega placement with GeoCalib background gravity before 3D fitting; manual calibration bypasses both. GeoCalib uses the base Flow support overlay and a shared checkpoint in Model Manager. Unreliable gravity needs manual calibration or clearer views; it does not fall back to the person as a vertical reference. Body and finger motion, original results and contact-corrected results are preserved. Best suited to views that complement each other; occlusion and bad masks can cause errors. Slower than single-video methods. Requires the separate Setup_MAMMA.cmd runtime and registered restricted-use models. See Model Manager > Runtime Setup and docs/MAMMA.md.',
    'gvhmr': 'Lowest overall motion quality in our tests, but sometimes better for unusual poses such as backflips. '
             'No native finger tracking. Optional HandFlow adds finger motion; FootMR refines ankles, and HTD refines body motion. These add processing time.',
    'flowhmr': 'Slow, but the best overall motion quality in our tests. Tracks body and fingers without a separate hand add-on. Motion generation uses 256-query attention batches with FP32 and full temporal context, reducing VRAM especially for long clips. Fast SAM offers independent Adaptive Batching, GPU Hand Crops and default-on 4-Frame Hand Sampling. Four-frame hands interpolate intermediate features: tests reduced warm feature time by 41-49% and allocated VRAM from 5.43 to 4.21 GiB at batch 16, but finger accuracy, body/wrist motion and automatic contacts can change. Turn it off for every-frame hand inference. Automatic selects 1-16 spatial frames; Low Memory uses one and retains hand sampling. Whole-job gains vary because loading dominates short clips. Query size, hand interval, batches and OOM retries appear in Console and CMD. Optional Refine Camera Rotation changes camera display only. See docs/models/FLOWHMR.md.',
    'exmr': 'Generally fast, with good motion quality in our tests. A practical choice for a first attempt. Tracks body and fingers together without a separate finger add-on. Optional Refine Camera Rotation validates small background-based corrections for camera display only; source motion is unchanged.',
}


DESCRIPTIONS['flowhmr'] += (' Optional HaMeR Finger Refinement aims to improve finger tracking accuracy and defaults off. It uses ExMR hand crops and Gaussian rotation smoothing, not ExMR learned HandMR inference. Only fingers are replaced; invalid crops retain FlowHMR hands. Slower and may still jitter. Enabling it also enables Fast SAM and 4-Frame Hand Sampling; HaMeR itself uses every frame. Shared HaMeR/MANO files are prepared in Model Manager, and the existing ExMR runtime runs this pass. Reconstruct 3D after changing it.')


def guide_html(selected):
    descriptions = ''.join(
        f'<p><b>{tool.label}</b> &nbsp; <a href="{PROJECT_URLS[key]}">Original project</a><br>'
        f'{DESCRIPTIONS[key]}</p>' for key, tool in MOTION_TOOLS.items())
    rows = ''.join(
        ('<tr bgcolor="#303f50">' if key == selected else '<tr>') +
        f'<td><b>{MOTION_TOOLS[key].label}</b></td>' +
        ''.join(f'<td>{value}</td>' for value in values) + '</tr>'
        for key, values in BENCHMARKS.items())
    return """<html><head><style>
body { font-size:12px; } h3 { margin-top:15px; } a { color:#8ed2ff; }
th { text-align:left; background-color:#333333; } td, th { padding:7px; }
</style></head><body>
<h3>Choosing a model</h3>
<p>These are subjective observations from our test videos, not a universal ranking or accuracy benchmark.
Try another model if a particular movement looks wrong.</p>""" + descriptions + """
<p>Offline source motion: File &gt; Export offers FBX, BVH and portable Blender projects with video/cameras. MAMMA also offers a whole-scene Blender bundle with checked body meshes and all cameras. Keep exported video/mesh folders together. No additional inference is run for export.</p>
<p><b>Resource Safety</b> defaults on below Resource Mode. On retains low-RAM stopping and the PyTorch VRAM allocation limit. Off bypasses these limits; Automatic batching, supported Low Memory paths and actual allocation-error handling remain. The preference is saved locally, separately from projects. PEAR inherits it while its offline resource controls are hidden. See docs/PERFORMANCE.md for details.</p>
<h3>Recorded processing time and memory</h3>
<table width="98%" border="0" cellspacing="1" cellpadding="7">
<tr><th>Model</th><th>Input</th><th>Processing time</th><th>Peak VRAM (GiB)</th><th>Peak RAM (GiB)</th></tr>""" + rows + """</table>
<p>Test PC: RTX 5090 32 GB · Ryzen 9 9950X · 96 GB RAM.</p>
<p>GVHMR: HTD + FootMR + HandFlow on. FlowHMR: original SAM; today's Fast SAM can
be faster. These three models used the same seven-second clip; their times exclude
tracking, app correction and preview/export.</p>
<p>MAMMA used a different clip: three synchronized videos resized within 1920×1080,
two people and 413 frames at 30 fps. Its time includes preparation, correction and preview;
memory covers masks, landmarks, VGGT and 3D fitting. This is not a direct speed comparison.
That row predates direct SAM video input. New paired four-view tests skipped intermediate
RGB PNGs and cut mask-stage time by 17–21% with nearly unchanged RAM/VRAM and identical
masks; this is not a whole-job speedup. See docs/MAMMA.md and docs/PERFORMANCE.md.</p>
<p>PromptHMR used 422 frames at 30 fps, one fixed-camera video and two people, with
Interaction Prompt and app correction on. Time includes tracking, native inference,
post-optimization, correction and preview/bone preparation; writing public FBX files is excluded.
Its peak reserved CUDA memory was 7.83 GiB. It used a different clip from the older three-model test.</p>
<p>PEAR is a live preview, not a full-clip reconstruction. Its row measures the fixed-length bone worker after warmup, excluding model startup, GUI and camera capture. Maximum processing time was 160.31 ms; the visible app presented 150 paired frames in five seconds. Startup took about 34 seconds. Peak RAM was not separately recorded for this row. See docs/models/PEAR.md for scope and limits. This is not directly comparable with offline processing times.</p>
<p>VRAM is peak PyTorch allocated memory, not total GPU usage. ExMR/FlowHMR/GVHMR RAM is
peak private memory; MAMMA/PromptHMR RAM is peak worker working set. The older four-model
measurements predate the latest Low Memory changes. These are not minimum requirements.
Results vary by input and PC.</p>
<p>Offline reconstruction models offer Motion Correction for foot and hand contacts. Contact correction reduces sliding;
it is separate from tracking fingers or refining ankle angles.</p></body></html>"""


class ModelGuideDialog(QDialog):
    def __init__(self, parent, selected):
        super().__init__(parent)
        self.setWindowTitle('Motion Model Guide')
        self.setMinimumSize(360, 260)
        area = self.screen().availableGeometry()
        self.resize(min(740, max(360, area.width()-80)),
                    min(540, max(260, area.height()-100)))
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setAccessibleName('Motion model descriptions and short-clip benchmarks')
        self.browser.setStyleSheet('QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:6px; }')
        self.browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(QDesktopServices.openUrl)
        self.browser.setHtml(guide_html(selected))
        layout.addWidget(self.browser)
        actions = QHBoxLayout(); actions.addStretch()
        close = QPushButton('Close'); close.clicked.connect(self.accept)
        close.setToolTip('Close the model guide.')
        actions.addWidget(close); layout.addLayout(actions)
