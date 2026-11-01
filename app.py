from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    from mocap.portable_startup import configure_portable_environment
    configure_portable_environment()
    from mocap.runtime_limits import configure_preview_cpu
    configure_preview_cpu()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--smoke-screenshot", type=Path)
    parser.add_argument("--smoke-run-analysis", action="store_true")
    parser.add_argument("--smoke-playback-seconds", type=float, default=0)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--smoke-camera-distance", type=float)
    parser.add_argument('--smoke-render-mode',choices=['raw','box','mixamo'],default='box',help=argparse.SUPPRESS)
    parser.add_argument('--smoke-bone-overlay',choices=['on','off'])
    args = parser.parse_args()
    # Give Windows a stable app identity so the taskbar uses the app icon.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("toyxyz.RGBMoCap")
        except (AttributeError, OSError):
            pass
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon, QSurfaceFormat
    from PySide6.QtCore import QTimer
    from mocap.gui import MainWindow
    from mocap.theme import configure_application
    # QOpenGLWidget's private FBO format does not configure the top-level
    # window's composition context. Avoid blocking input on that context's
    # default swap interval; camera redraws are paced in MotionView.
    surface = QSurfaceFormat.defaultFormat()
    surface.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(surface)
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "rgbmocap-icon.png")))
    configure_application(app)
    window = MainWindow(args.video)
    window.bone_overlay.setChecked(args.smoke_bone_overlay=='on' if args.smoke_bone_overlay else args.smoke_render_mode!='raw')
    if args.project:
        window.load_project(args.project)
    window.show()
    if args.smoke_camera_distance is not None:
        window.motion_view.auto_fit = False
        window.motion_view.distance = max(0.4, min(80, args.smoke_camera_distance))
        window.motion_view.update()
    app.aboutToQuit.connect(window.shutdown_preview)
    if args.smoke_playback_seconds:
        from mocap.playback_check import schedule_check
        schedule_check(window, args.smoke_playback_seconds, args.smoke_report or Path("playback_report.json"),
                       args.smoke_screenshot, app.quit)
    if args.smoke_screenshot and not args.smoke_run_analysis and not args.smoke_playback_seconds:
        def capture(success=True):
            args.smoke_screenshot.parent.mkdir(parents=True, exist_ok=True)
            from mocap.capture import capture_window
            capture_window(window).save(str(args.smoke_screenshot))
            app.exit(0 if success else 1)
        if args.smoke_report:
            from mocap.startup_check import schedule_check
            QTimer.singleShot(400, lambda: schedule_check(window, args.smoke_report, capture))
        else:
            QTimer.singleShot(800, capture)
    if args.smoke_run_analysis:
        import json
        from mocap.storage import atomic_json
        deadline = __import__('time').monotonic() + 180
        watchdog = QTimer()
        watchdog.setInterval(250)
        def verify():
            if not window.busy or __import__('time').monotonic() > deadline:
                success = not window.busy and not window.failed and window.project is not None and (window.project / 'keypoints.npz').exists()
                if window.busy:
                    window.cancel_job()
                if args.smoke_screenshot:
                    args.smoke_screenshot.parent.mkdir(parents=True, exist_ok=True)
                    window.grab().save(str(args.smoke_screenshot))
                    atomic_json(args.smoke_screenshot.with_suffix('.json'), {'success': success, 'status': window.status.text(), 'project': str(window.project)})
                app.exit(0 if success else 1)
        def start_test():
            window.run_analysis()
            watchdog.timeout.connect(verify)
            watchdog.start()
        QTimer.singleShot(200, start_test)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
