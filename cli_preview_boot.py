"""Render through the desktop preview pipeline in an owned Qt process."""
import argparse
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser()
parser.add_argument('project', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--frames', type=int)
parser.add_argument('--camera', choices=('saved', 'refined'))
args = parser.parse_args()

from mocap.runtime_registry import activate
activate(root, 'gui')
from mocap.runtime_limits import configure_preview_cpu
configure_preview_cpu()
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from mocap.gui import MainWindow
from mocap.theme import configure_application, load_appearance

surface = QSurfaceFormat.defaultFormat()
surface.setSwapInterval(0)
QSurfaceFormat.setDefaultFormat(surface)
app = QApplication([sys.argv[0]])
configure_application(app, *load_appearance())
window = MainWindow(None)
window.load_project(args.project)
if args.camera:
    camera_tools = window.camera_tools
    camera_tools.restore_background({window.model_choice.currentData(): args.camera == 'refined'})
    window.viewport_checks['show_cameras'].setChecked(True)
    camera_tools.sync()
window.move(-20000, -20000)
window.show()
app.aboutToQuit.connect(window.shutdown_preview)
timer = QTimer()
timer.setInterval(100)
started, exporter = time.monotonic(), None


def tick():
    global exporter
    try:
        if exporter is None:
            if not window.preview_export_available() or not window.motion_view.isValid():
                if time.monotonic() - started > 30:
                    raise RuntimeError('The motion or desktop OpenGL context did not become available.')
                return
            if args.camera:
                data = camera_tools.data
                if not data or data.get('background_pending'):
                    timeout = 225 if args.camera == 'refined' else 30
                    if time.monotonic() - started > timeout:
                        raise RuntimeError(camera_tools.message or 'Saved camera preview did not become ready.')
                    return
                if report := data.get('background_report'):
                    summary = report.get('skipped') or report.get('failed') or data.get('note', '')
                    print(summary, flush=True)
            if args.frames is not None:
                first = window.slider.clip_in
                window.slider.set_clip(first, min(window.slider.clip_out, first+args.frames))
            window.start_preview_export(args.output, resume=False)
            exporter = window._preview_export
            exporter.dialog.move(-20000, -20000)
        elif window._preview_export is None:
            status = window.status.text()
            print(status, flush=True)
            app.exit(0 if args.output.is_file() and status.startswith('Preview saved:') else 1)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        app.exit(1)


timer.timeout.connect(tick)
timer.start()
raise SystemExit(app.exec())
