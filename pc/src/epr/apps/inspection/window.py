"""G1 inspection window: click RGB, see the matching thermal region."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from epr.core.domain_models.frame_set import Channel, FrameSet
from epr.processing.display import colorize_thermal, draw_probe_marks
from epr.processing.registration import ThermalProbe, probe


class ImagePane(QLabel):
    """Shows an RGB image and reports clicks as normalized coordinates."""

    clicked = Signal(float, float)

    def __init__(self, title: str) -> None:
        super().__init__()
        self._image: np.ndarray | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111; color: #eee;")
        self.setText(title)
        self.setMouseTracking(False)

    def set_image(self, image: np.ndarray) -> None:
        self._image = np.ascontiguousarray(image)
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def mousePressEvent(self, event) -> None:  # noqa: N802
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
        pixmap = QPixmap.fromImage(qimage)
        fitted = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(fitted)


class InspectionWindow(QMainWindow):
    def __init__(self, frame_sets: Sequence[FrameSet], *, title: str = "EPR inspection") -> None:
        super().__init__()
        if not frame_sets:
            raise ValueError("no frame sets to inspect")
        self._sets = list(frame_sets)
        self._index = 0
        self.setWindowTitle(title)
        self.resize(1100, 640)

        self._chooser = QComboBox()
        for frame_set in self._sets:
            self._chooser.addItem(f"frame set {frame_set.frame_set_id}")
        self._chooser.currentIndexChanged.connect(self._change_frame)

        self._rgb = ImagePane("RGB")
        self._thermal = ImagePane("Thermal")
        self._rgb.clicked.connect(self._on_click)

        self._status = QLabel("Click a point on the RGB image.")
        self._status.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self._rgb, 1)
        row.addWidget(self._thermal, 1)

        root = QVBoxLayout()
        root.addWidget(self._chooser)
        root.addLayout(row, 1)
        root.addWidget(self._status)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self._show_frame()

    def _change_frame(self, index: int) -> None:
        self._index = index
        self._show_frame()

    def _show_frame(self, mark: ThermalProbe | None = None) -> None:
        frame_set = self._sets[self._index]
        rgb = frame_set.array(Channel.RGB)
        thermal = colorize_thermal(frame_set.array(Channel.THERMAL))
        if mark is not None:
            rgb = draw_probe_marks(rgb, mark.rgb_pixel, (0, 255, 80))
            thermal = draw_probe_marks(thermal, mark.thermal_pixel, (255, 255, 255))
        self._rgb.set_image(rgb)
        self._thermal.set_image(thermal)
        if mark is None:
            meta = frame_set.metadata
            self._status.setText(
                f"frame set {frame_set.frame_set_id}  ·  "
                f"ambient {meta.environment.ambient_temperature_c:.1f} C  ·  "
                f"distance {meta.geometry.distance_mm:.0f} mm  ·  "
                "click RGB to probe thermal"
            )

    def _on_click(self, x: float, y: float) -> None:
        mark = probe(self._sets[self._index], x, y)
        self._show_frame(mark)
        self._status.setText(
            f"RGB ({mark.rgb_pixel[0]}, {mark.rgb_pixel[1]})  →  "
            f"thermal ({mark.thermal_pixel[0]}, {mark.thermal_pixel[1]})  ·  "
            f"T {mark.temperature_c:.1f} C  ·  "
            f"region mean {mark.region_mean_c:.1f} C  max {mark.region_max_c:.1f} C"
        )
