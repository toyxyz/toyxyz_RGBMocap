"""Playback at the end of a trimmed clip, using the real timeline controls."""
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from mocap.gui import MainWindow


APP = QApplication.instance() or QApplication([])


class TimelinePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.window = w = MainWindow()
        self.addCleanup(w.close)
        w.timer_resolution = Mock()
        w.start_preview_audio = Mock()
        w.preview_audio = Mock()
        w.play_clock = Mock()
        w.play_clock.nsecsElapsed.return_value = 0
        w.video_path = Path('test-preview.mp4')
        w.video_readable = True
        w.info = {'fps': 30.0, 'duration': 1.0}
        w.source_end = 1.0
        w.slider.setRange(0, 29)
        w.slider.set_clip(3, 9)
        w.slider.setValue(3)
        w.update_readiness()
        w.preview_audio.reset_mock()

    def tick(self, steps):
        w = self.window
        rate = w.info['fps'] * w.play_speed.currentData()
        w.play_clock.nsecsElapsed.return_value = math.ceil(steps * 1e9 / rate)
        w.next_frame()

    def test_loop_is_on_by_default(self):
        w = self.window
        self.assertTrue(w.loop_playback.isChecked())
        w.play_button.click()
        self.tick(6)
        self.assertEqual(w.slider.value(), 3)
        self.assertTrue(w.timer.isActive())
        w.preview_audio.sync.assert_called_once_with(3 / 30, looping=True)

    def test_one_pass_holds_clip_end_even_if_a_tick_skips_it(self):
        w = self.window
        w.loop_playback.click()
        w.play_speed.setCurrentIndex(w.play_speed.findData(2.0))
        w.slider.setValue(6)
        w.play_button.click()
        self.tick(5)
        self.assertEqual(w.slider.value(), 8)
        self.assertFalse(w.timer.isActive())
        self.assertEqual(w.play_button.accessibleName(), 'Play')
        w.preview_audio.pause.assert_called_once()
        w.play_button.click()
        self.assertEqual(w.slider.value(), 3)

    def test_disabling_loop_during_playback_finishes_the_current_pass(self):
        w = self.window
        w.play_button.click()
        self.tick(8)
        self.assertEqual(w.slider.value(), 5)
        w.loop_playback.click()
        self.assertEqual(w.play_anchor, 5)
        self.tick(3)
        self.assertEqual(w.slider.value(), 8)
        self.assertTrue(w.timer.isActive())
        self.tick(4)
        self.assertEqual(w.slider.value(), 8)
        self.assertFalse(w.timer.isActive())
