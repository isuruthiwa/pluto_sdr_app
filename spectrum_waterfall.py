import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QFont, QTransform

WATERFALL_ROWS = 120


def _make_colormap() -> pg.ColorMap:
    """Inferno-like colormap for waterfall."""
    pos = np.linspace(0, 1, 7)
    colors = np.array([
        [  0,   0,   0, 255],
        [ 40,   0,  80, 255],
        [120,   0, 120, 255],
        [200,  30,  60, 255],
        [255, 120,   0, 255],
        [255, 220,   0, 255],
        [255, 255, 255, 255],
    ], dtype=np.uint8)
    return pg.ColorMap(pos, colors)


class SpectrumWaterfallWidget(QWidget):
    """
    Combined spectrum (top) + waterfall (bottom) display.
    Emits freq_clicked(Hz) when user left-clicks anywhere in either panel.
    """

    freq_clicked = pyqtSignal(float)   # absolute Hz

    def __init__(self, sample_rate: float = 2e6, center_freq: float = 100e6,
                 parent=None):
        super().__init__(parent)
        self.sample_rate  = sample_rate
        self.center_freq  = center_freq
        self.fft_size     = 2048
        self._rx_offset   = 0.0   # Hz offset of receive channel from center

        # Rolling waterfall buffer: shape (ROWS, fft_size)
        self._wf_buf = np.full((WATERFALL_ROWS, self.fft_size),
                               -120.0, dtype=np.float32)

        self._build_ui()
        self._update_transforms()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground('#0d0d1a')
        layout.addWidget(self._glw)

        tick_font = QFont('Consolas', 7)
        axis_pen  = pg.mkPen('#445')

        # ── Spectrum plot ────────────────────────────────────────────────────
        self._sp = self._glw.addPlot(row=0, col=0)
        self._sp.setLabel('left',   'Power', units='dBFS',
                          **{'color': '#aaa', 'font-size': '8pt'})
        self._sp.setLabel('bottom', 'Frequency', units='MHz',
                          **{'color': '#aaa', 'font-size': '8pt'})
        self._sp.setYRange(-110, -10, padding=0)
        self._sp.showGrid(x=True, y=True, alpha=0.25)
        self._sp.getAxis('left').setStyle(tickFont=tick_font)
        self._sp.getAxis('bottom').setStyle(tickFont=tick_font)
        self._sp.setMouseEnabled(x=False, y=True)
        self._sp.setMenuEnabled(False)
        self._sp.setDownsampling(auto=True, mode='peak')
        self._sp.setClipToView(True)

        # FFT curve
        self._curve = self._sp.plot(
            pen=pg.mkPen(color='#00c8ff', width=1),
            antialias=False,
        )

        # Filled area under curve
        self._fill_curve = self._sp.plot(
            pen=pg.mkPen(color='#00c8ff', width=1),
            fillLevel=-120,
            brush=(0, 180, 255, 30),
            antialias=False,
        )

        # Rx channel marker (vertical line)
        self._rx_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color='#ffcc00', width=1,
                         style=Qt.DashLine),
            label='RX', labelOpts={'color': '#ffcc00', 'movable': False},
        )
        self._sp.addItem(self._rx_line)

        # ── Waterfall plot ───────────────────────────────────────────────────
        self._wf = self._glw.addPlot(row=1, col=0)
        self._wf.setLabel('left', 'Time', **{'color': '#aaa', 'font-size': '8pt'})
        self._wf.getAxis('left').setStyle(showValues=False)
        self._wf.getAxis('bottom').setStyle(tickFont=tick_font)
        self._wf.setMouseEnabled(x=False, y=False)
        self._wf.setMenuEnabled(False)
        self._wf.setXLink(self._sp)

        cmap = _make_colormap()
        self._wf_img = pg.ImageItem()
        self._wf_img.setColorMap(cmap)
        self._wf_img.setLevels([-110, -20])
        self._wf.addItem(self._wf_img)

        # Rx marker on waterfall
        self._wf_rx_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color='#ffcc00', width=1, style=Qt.DashLine),
        )
        self._wf.addItem(self._wf_rx_line)

        # Size ratio: spectrum 1 part, waterfall 1 part
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 1)

        # Mouse clicks
        self._sp.scene().sigMouseClicked.connect(self._on_click)
        self._wf.scene().sigMouseClicked.connect(self._on_click)

    # ── public update API ────────────────────────────────────────────────────

    def update_fft(self, freqs: np.ndarray, power_db: np.ndarray):
        """
        freqs: relative Hz (0-centered), length = fft_size
        power_db: dBFS values, same length
        """
        abs_mhz = (freqs + self.center_freq) / 1e6

        self._curve.setData(x=abs_mhz, y=power_db)
        self._fill_curve.setData(x=abs_mhz, y=power_db)

        # Waterfall: roll buffer, add new row at bottom
        self._wf_buf = np.roll(self._wf_buf, -1, axis=0)
        self._wf_buf[-1] = power_db

        # Display: transpose so x=freq, y=time; newest row at bottom (y=0)
        self._wf_img.setImage(
            np.flipud(self._wf_buf).T,
            autoLevels=False,
        )

    def set_center_freq(self, freq_hz: float):
        self.center_freq = freq_hz
        self._update_transforms()

    def set_sample_rate(self, rate: float):
        self.sample_rate = rate
        self._update_transforms()

    def set_rx_offset(self, offset_hz: float):
        """Move the receive-channel marker."""
        self._rx_offset = offset_hz
        rx_mhz = (self.center_freq + offset_hz) / 1e6
        self._rx_line.setPos(rx_mhz)
        self._wf_rx_line.setPos(rx_mhz)

    def set_power_range(self, min_db: float, max_db: float):
        self._sp.setYRange(min_db, max_db, padding=0)
        self._wf_img.setLevels([min_db, max_db])

    # ── internals ────────────────────────────────────────────────────────────

    def _update_transforms(self):
        lo_mhz  = (self.center_freq - self.sample_rate / 2) / 1e6
        hi_mhz  = (self.center_freq + self.sample_rate / 2) / 1e6
        span    = hi_mhz - lo_mhz

        self._sp.setXRange(lo_mhz, hi_mhz, padding=0)

        # Map image pixels to frequency axis
        tr = QTransform()
        tr.translate(lo_mhz, 0)
        tr.scale(span / self.fft_size, 1.0)
        self._wf_img.setTransform(tr)

        # Reposition rx marker
        self.set_rx_offset(self._rx_offset)

    def _on_click(self, event):
        """Convert scene click → absolute frequency → emit signal."""
        if event.button() != Qt.LeftButton:
            return
        for plot in (self._sp, self._wf):
            if plot.sceneBoundingRect().contains(event.scenePos()):
                pt = plot.vb.mapSceneToView(event.scenePos())
                self.freq_clicked.emit(pt.x() * 1e6)
                return
