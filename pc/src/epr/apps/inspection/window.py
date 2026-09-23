"""G1 inspection window: click RGB, see thermal region, part name and heat verdict."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from epr.core.domain_models.component import ThermalCondition
from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.core.domain_models.region import ComponentRegion
from epr.processing.display import colorize_thermal, draw_probe_marks, draw_region_overlays
from epr.processing.inspection import InspectionHit, inspect_click

STYLESHEET = """
QMainWindow, QWidget#root {
    background-color: #0f1419;
    color: #e7eef5;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QLabel#title {
    font-size: 18px;
    font-weight: 600;
    color: #f4f7fb;
}
QLabel#subtitle {
    color: #8fa3b8;
    font-size: 12px;
}
QFrame#pane, QFrame#side {
    background-color: #171e26;
    border: 1px solid #2a3542;
    border-radius: 10px;
}
QLabel#paneCaption {
    color: #8fa3b8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 12px 0 12px;
}
QLabel#imagePane {
    background-color: #0b1015;
    border-radius: 0 0 10px 10px;
    min-height: 320px;
}
QLabel#imagePane[hovering="true"] {
    border: 1px solid #3dd6c6;
}
QComboBox, QPushButton {
    background-color: #222c38;
    color: #e7eef5;
    border: 1px solid #354556;
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 18px;
}
QComboBox:hover, QPushButton:hover {
    background-color: #2b3848;
    border-color: #3dd6c6;
}
QPushButton:pressed {
    background-color: #1a222c;
}
QPushButton#primary {
    background-color: #1f6f78;
    border-color: #3dd6c6;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #25848e;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #2a3542;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    background: #3dd6c6;
    border-radius: 8px;
}
QLabel#metricName {
    color: #8fa3b8;
    font-size: 11px;
}
QLabel#metricValue {
    color: #f4f7fb;
    font-size: 22px;
    font-weight: 600;
}
QLabel#metricUnit {
    color: #8fa3b8;
    font-size: 12px;
}
QLabel#verdict {
    border-radius: 8px;
    padding: 10px 12px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.04em;
}
QLabel#reason {
    color: #b7c7d6;
    font-size: 12px;
}
QLabel#hint {
    color: #6f8499;
    font-size: 11px;
}
"""

VERDICT_STYLE = {
    ThermalCondition.NORMAL: ("#10261c", "#3dd68c", "NORMAL"),
    ThermalCondition.ELEVATED: ("#2a220e", "#f0b429", "ELEVATED"),
    ThermalCondition.HOT: ("#2a160e", "#ff7a45", "HOT"),
    ThermalCondition.ABNORMAL_RISE: ("#2a1014", "#ff5c7a", "ABNORMAL"),
    ThermalCondition.UNSTABLE: ("#22102a", "#c084fc", "UNSTABLE"),
    ThermalCondition.UNKNOWN: ("#1a222c", "#8fa3b8", "UNKNOWN"),
}


class ImagePane(QLabel):
    """Shows an RGB image and reports clicks as normalized coordinates."""

    clicked = Signal(float, float)
    hovered = Signal(float, float)
    left = Signal()

    def __init__(self, title: str) -> None:
        super().__init__()
        self._image: np.ndarray | None = None
        self._title = title
        self.setObjectName("imagePane")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText(title)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: np.ndarray) -> None:
        self._image = np.ascontiguousarray(image)
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hovering", True)
        self.style().unpolish(self)
        self.style().polish(self)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hovering", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.left.emit()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        mapped = self._widget_to_normalized(event.position().x(), event.position().y())
        if mapped is not None:
            self.hovered.emit(*mapped)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._image is None or not self.pixmap() or self.pixmap().isNull():
            return
        mapped = self._widget_to_normalized(event.position().x(), event.position().y())
        if mapped is not None:
            self.clicked.emit(*mapped)

    def _widget_to_normalized(self, x: float, y: float) -> tuple[float, float] | None:
        pixmap = self.pixmap()
        if pixmap is None or self._image is None:
            return None
        offset_x = (self.width() - pixmap.width()) / 2
        offset_y = (self.height() - pixmap.height()) / 2
        px = x - offset_x
        py = y - offset_y
        if px < 0 or py < 0 or px >= pixmap.width() or py >= pixmap.height():
            return None
        return px / pixmap.width(), py / pixmap.height()

    def _refresh(self) -> None:
        if self._image is None:
            return
        height, width = self._image.shape[:2]
        qimage = QImage(self._image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        fitted = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(fitted)


class MetricBlock(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = QLabel(name)
        self._name.setObjectName("metricName")
        self._value = QLabel("—")
        self._value.setObjectName("metricValue")
        self._unit = QLabel("")
        self._unit.setObjectName("metricUnit")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._name)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._value)
        row.addWidget(self._unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        layout.addLayout(row)

    def set_value(self, value: str, unit: str = "") -> None:
        self._value.setText(value)
        self._unit.setText(unit)


class InspectionWindow(QMainWindow):
    def __init__(
        self,
        frame_sets: Sequence[FrameSet],
        *,
        regions: Sequence[ComponentRegion] = (),
        title: str = "EPR inspection",
    ) -> None:
        super().__init__()
        if not frame_sets:
            raise ValueError("no frame sets to inspect")
        self._sets = list(frame_sets)
        self._regions = list(regions)
        self._index = len(self._sets) - 1
        self._hit: InspectionHit | None = None
        self._show_overlays = True
        self.setWindowTitle(title)
        self.resize(1280, 760)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())
        layout.addLayout(self._build_toolbar())
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._build_viewers(), 3)
        body.addWidget(self._build_side_panel(), 1)
        layout.addLayout(body, 1)
        layout.addWidget(self._build_hint())

        self._bind_shortcuts()
        self._chooser.setCurrentIndex(self._index)
        self._slider.setValue(self._index)
        self._show_frame()

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Board inspection")
        title.setObjectName("title")
        subtitle = QLabel("Click a part on the RGB image to read temperature and heat verdict")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        return header

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._prev = QPushButton("← Prev")
        self._next = QPushButton("Next →")
        self._prev.clicked.connect(lambda: self._step_frame(-1))
        self._next.clicked.connect(lambda: self._step_frame(1))

        self._chooser = QComboBox()
        for frame_set in self._sets:
            elapsed = frame_set.metadata.timestamp_ns / 1e9
            self._chooser.addItem(f"Frame {frame_set.frame_set_id}  ·  t = {elapsed:.1f} s")
        self._chooser.currentIndexChanged.connect(self._change_frame)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(self._sets) - 1)
        self._slider.valueChanged.connect(self._change_frame)

        self._overlay_btn = QPushButton("Hide outlines")
        self._overlay_btn.setCheckable(True)
        self._overlay_btn.clicked.connect(self._toggle_overlays)

        self._clear_btn = QPushButton("Clear probe")
        self._clear_btn.clicked.connect(self._clear_probe)

        bar.addWidget(self._prev)
        bar.addWidget(self._next)
        bar.addWidget(self._chooser, 1)
        bar.addWidget(self._slider, 2)
        bar.addWidget(self._overlay_btn)
        bar.addWidget(self._clear_btn)
        return bar

    def _build_viewers(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        self._rgb = ImagePane("RGB")
        self._thermal = ImagePane("Thermal")
        self._rgb.clicked.connect(self._on_click)
        self._rgb.hovered.connect(self._on_hover)
        self._rgb.left.connect(lambda: self._hover.setText("Hover the board, then click a part"))

        row.addWidget(self._wrap_pane("RGB board", self._rgb), 1)
        row.addWidget(self._wrap_pane("Thermal map", self._thermal), 1)
        return row

    def _wrap_pane(self, caption: str, pane: ImagePane) -> QFrame:
        frame = QFrame()
        frame.setObjectName("pane")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(caption)
        label.setObjectName("paneCaption")
        layout.addWidget(label)
        layout.addWidget(pane, 1)
        return frame

    def _build_side_panel(self) -> QFrame:
        side = QFrame()
        side.setObjectName("side")
        side.setMinimumWidth(280)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self._part = QLabel("No part selected")
        self._part.setObjectName("title")
        self._part.setWordWrap(True)
        layout.addWidget(self._part)

        self._package = QLabel("Click the RGB board to identify a component")
        self._package.setObjectName("subtitle")
        self._package.setWordWrap(True)
        layout.addWidget(self._package)

        self._verdict = QLabel("WAITING")
        self._verdict.setObjectName("verdict")
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_verdict(None)
        layout.addWidget(self._verdict)

        self._reason = QLabel("")
        self._reason.setObjectName("reason")
        self._reason.setWordWrap(True)
        layout.addWidget(self._reason)

        self._t = MetricBlock("Temperature")
        self._dt = MetricBlock("Rise above ambient")
        self._rate = MetricBlock("Heating rate")
        layout.addWidget(self._t)
        layout.addWidget(self._dt)
        layout.addWidget(self._rate)

        self._meta = QLabel("")
        self._meta.setObjectName("subtitle")
        self._meta.setWordWrap(True)
        layout.addWidget(self._meta)
        layout.addStretch(1)

        self._hover = QLabel("Hover the board, then click a part")
        self._hover.setObjectName("hint")
        self._hover.setWordWrap(True)
        layout.addWidget(self._hover)
        return side

    def _build_hint(self) -> QLabel:
        hint = QLabel(
            "Shortcuts: ← → change frame · O toggle outlines · Esc clear probe · "
            "click RGB to probe thermal"
        )
        hint.setObjectName("hint")
        return hint

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step_frame(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step_frame(1))
        QShortcut(QKeySequence(Qt.Key.Key_O), self, self._overlay_btn.click)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._clear_probe)

    def _step_frame(self, delta: int) -> None:
        self._change_frame(min(max(self._index + delta, 0), len(self._sets) - 1))

    def _change_frame(self, index: int) -> None:
        if index < 0 or index >= len(self._sets):
            return
        self._index = index
        if self._chooser.currentIndex() != index:
            self._chooser.blockSignals(True)
            self._chooser.setCurrentIndex(index)
            self._chooser.blockSignals(False)
        if self._slider.value() != index:
            self._slider.blockSignals(True)
            self._slider.setValue(index)
            self._slider.blockSignals(False)
        self._prev.setEnabled(index > 0)
        self._next.setEnabled(index < len(self._sets) - 1)
        # Keep the last click when scrubbing frames so heating can be watched.
        if self._hit is not None:
            x, y = self._hit.mark.rgb_xy
            self._on_click(x, y)
        else:
            self._show_frame()

    def _toggle_overlays(self) -> None:
        self._show_overlays = not self._overlay_btn.isChecked()
        label = "Show outlines" if self._overlay_btn.isChecked() else "Hide outlines"
        self._overlay_btn.setText(label)
        self._show_frame(self._hit.mark if self._hit else None)

    def _clear_probe(self) -> None:
        self._hit = None
        self._show_frame()
        self._reset_side_panel()

    def _previous(self) -> FrameSet | None:
        return self._sets[self._index - 1] if self._index > 0 else None

    def _show_frame(self, mark=None) -> None:
        frame_set = self._sets[self._index]
        rgb = frame_set.array(Channel.RGB)
        thermal = colorize_thermal(frame_set.array(Channel.THERMAL))
        active_id = self._hit.region.region_id if self._hit and self._hit.region else None
        if self._show_overlays and self._regions:
            rgb = draw_region_overlays(rgb, self._regions, active_id=active_id)
        if mark is not None:
            label = self._hit.part_label if self._hit else None
            rgb = draw_probe_marks(rgb, mark.rgb_pixel, (61, 214, 198), label=label)
            thermal = draw_probe_marks(thermal, mark.thermal_pixel, (255, 255, 255))
        self._rgb.set_image(rgb)
        self._thermal.set_image(thermal)

        meta = frame_set.metadata
        self._meta.setText(
            f"Frame {frame_set.frame_set_id} · "
            f"ambient {meta.environment.ambient_temperature_c:.1f} °C · "
            f"distance {meta.geometry.distance_mm:.0f} mm · "
            f"{len(self._regions)} annotated parts"
        )
        if self._hit is None:
            self._reset_side_panel()

    def _reset_side_panel(self) -> None:
        self._part.setText("No part selected")
        self._package.setText("Click the RGB board to identify a component")
        self._reason.setText("")
        self._t.set_value("—")
        self._dt.set_value("—")
        self._rate.set_value("—")
        self._set_verdict(None)

    def _set_verdict(self, condition: ThermalCondition | None) -> None:
        key = condition or ThermalCondition.UNKNOWN
        background, foreground, text = VERDICT_STYLE[key]
        if condition is None:
            text = "WAITING"
        self._verdict.setText(text)
        self._verdict.setStyleSheet(
            f"background-color: {background}; color: {foreground}; "
            "border-radius: 8px; padding: 10px 12px; font-weight: 700; font-size: 14px;"
        )

    def _on_hover(self, x: float, y: float) -> None:
        self._hover.setText(f"Cursor  {x:.3f}, {y:.3f}  ·  click to probe")

    def _on_click(self, x: float, y: float) -> None:
        hit = inspect_click(
            self._sets[self._index],
            x,
            y,
            regions=self._regions,
            previous=self._previous(),
        )
        self._hit = hit
        self._show_frame(hit.mark)

        self._part.setText(hit.part_label)
        if hit.region is not None:
            bits = []
            if hit.region.component_class:
                bits.append(hit.region.component_class)
            if hit.region.package:
                bits.append(hit.region.package)
            self._package.setText(" · ".join(bits) if bits else hit.region.region_id)
        else:
            self._package.setText("No annotated region under this click")

        self._t.set_value(f"{hit.mark.temperature_c:.1f}", "°C")
        self._dt.set_value(f"{hit.metrics.delta_t_c:.1f}", "°C")
        self._rate.set_value(f"{hit.metrics.heating_rate_c_s:.2f}", "°C/s")

        if hit.assessment is not None:
            self._set_verdict(hit.assessment.condition)
            self._reason.setText(hit.assessment.reason)
        else:
            self._set_verdict(None)
            self._reason.setText("")
