"""Play source audio against the existing frame-based preview clock."""
from pathlib import Path
import time

from PySide6.QtCore import QObject, QUrl


def source_for_preview(window):
    source = window.video_path
    start = window.source_offset
    if window.model_choice.currentData() == 'mamma' and window.mamma_inputs:
        source = window.mamma_inputs[window.mamma_active_camera]['path']
    elif not window._pear_active:
        if window.workspace_root is not None:
            from .project_workspace import read
            start -= float(read(window.workspace_root).get('time_offset', 0.))
        elif window.project is not None:
            from .project_storage import manifest
            start -= float(manifest(window.project).get('storage', {}).get('time_offset', 0.))
    if source is not None and Path(source).is_file():
        return Path(source), max(0., start)
    decoder = window.video_view.decoder
    if decoder is not None and decoder.path and Path(decoder.path).is_file():
        return Path(decoder.path), window.video_view.timeline_start / window.info['fps']
    return None, 0.


class PreviewAudio(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        self.output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.mediaStatusChanged.connect(self._apply_position)
        self.player.seekableChanged.connect(self._apply_position)
        self.active = self.pending = False
        self.timeline_start = 0.
        self.position_ms = 0
        self.last_seek = 0.
        self.lead_ms = 0

    def start(self, path, timeline_start, seconds, speed):
        source = QUrl.fromLocalFile(str(path.resolve()))
        if source != self.player.source():
            self.player.setSource(source)
        self.timeline_start = timeline_start
        self.active = True
        self.reanchor(seconds, speed)

    def pause(self):
        self.active = self.pending = False
        if self.player.playbackState() == self.player.PlaybackState.PlayingState:
            self.player.pause()

    def close(self):
        self.player.setSource(QUrl())

    def reanchor(self, seconds, speed):
        self.player.setPlaybackRate(speed)
        self.sync(seconds, looping=True)

    def sync(self, seconds, looping=False):
        if not self.active:
            return
        self.position_ms = round((self.timeline_start + seconds) * 1000)
        if looping:
            self.pending = True
        if self.pending:
            self._apply_position()
        elif (self.player.playbackState() == self.player.PlaybackState.PlayingState
                and time.monotonic() - self.last_seek >= 1.
                and abs(self.player.position() - self.position_ms) >= 180):
            # A visible frame may drop while decoding. Only correct sustained
            # clock drift or a late audio startup, never seek for every frame.
            if time.monotonic() - self.last_seek < 3.:
                # The previous seek left this lag (slow seek or output latency), so repeating
                # it would stutter forever. Lead later seeks by the lag instead.
                self.lead_ms = max(0, min(500, self.lead_ms + self.position_ms - self.player.position()))
            self.pending = True
            self._apply_position()

    def _apply_position(self, *_):
        player = self.player
        if not self.active or not self.pending or not player.isSeekable():
            return
        if player.mediaStatus() not in (player.MediaStatus.LoadedMedia, player.MediaStatus.BufferingMedia,
                                        player.MediaStatus.BufferedMedia, player.MediaStatus.EndOfMedia):
            return
        self.pending = False
        if not player.hasAudio() or self.position_ms >= player.duration():
            if player.playbackState() == player.PlaybackState.PlayingState:
                player.pause()
            return
        self.last_seek = time.monotonic()
        player.setPosition(self.position_ms + self.lead_ms)
        player.play()
