import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QLabel, QLineEdit, QPushButton,
    QSlider, QComboBox, QGroupBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QSizePolicy, QFrame,
    QPlainTextEdit, QFileDialog,
    QStatusBar, QAction, QMenu, QMenuBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from spectrum_waterfall import SpectrumWaterfallWidget
from sdr_worker import SDRWorker
from audio_worker import AudioWorker

# Supported demodulation modes
DEMOD_MODES = ['nfm', 'wfm', 'am', 'usb', 'lsb']

# Preset bandwidths per mode (Hz)
BW_PRESETS = {
    'nfm': 12_500,
    'wfm': 200_000,
    'am':  10_000,
    'usb':  3_000,
    'lsb':  3_000,
}

SAMPLE_RATES = [
    ('0.5 MHz',  500_000),
    ('1 MHz',  1_000_000),
    ('2 MHz',  2_000_000),
    ('4 MHz',  4_000_000),
    ('5 MHz',  5_000_000),
]


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setFont(QFont('Segoe UI', 8, QFont.Bold))
    return g


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont('Segoe UI', 8))
    return lbl


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySDR Receiver")
        self.resize(1280, 800)

        # Workers
        self._sdr   = SDRWorker()
        self._audio = AudioWorker()

        self._center_freq = self._sdr.center_freq
        self._rx_offset   = 0.0
        self._demod_mode  = 'nfm'

        self._build_ui()
        self._connect_signals()

        # Status refresh timer
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(500)

        # Auto-start in simulation mode
        self._sdr.try_connect()
        self._sdr.start()
        self._audio.start()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        # Central display with dedicated bottom TX/RX bar
        self._display = SpectrumWaterfallWidget(
            sample_rate=self._sdr.sample_rate,
            center_freq=self._sdr.center_freq,
        )

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._display)
        central_layout.addWidget(self._build_bottom_bar())
        self.setCentralWidget(central)

        # Controls dock
        dock = QDockWidget("Controls", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(320)
        dock.setMaximumWidth(420)
        dock.setWidget(self._build_controls())
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Menu bar
        self._build_menu()

        # Status bar
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb_left  = QLabel("Disconnected")
        self._sb_right = QLabel("")
        self._sb.addWidget(self._sb_left, 1)
        self._sb.addPermanentWidget(self._sb_right)

    def _build_controls(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(6)
        vbox.setContentsMargins(6, 6, 6, 6)

        # ── Connection ───────────────────────────────────────────────────────
        grp = _group("Connection")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(_label("URI:"))
        self._uri_edit = QLineEdit("ip:192.168.2.1")
        self._uri_edit.setFont(QFont('Consolas', 8))
        row.addWidget(self._uri_edit)
        gl.addLayout(row)

        self._conn_btn = QPushButton("Connect")
        self._conn_btn.setCheckable(True)
        gl.addWidget(self._conn_btn)

        self._sim_label = QLabel("● Simulation mode")
        self._sim_label.setStyleSheet("color:#f90; font-size:8pt;")
        gl.addWidget(self._sim_label)
        vbox.addWidget(grp)

        # ── Center Frequency ─────────────────────────────────────────────────
        grp = _group("Center Frequency")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(70.0, 6000.0)
        self._freq_spin.setDecimals(4)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setValue(self._center_freq / 1e6)
        self._freq_spin.setSingleStep(0.1)
        self._freq_spin.setFont(QFont('Consolas', 10, QFont.Bold))
        self._freq_spin.setMinimumHeight(28)
        row.addWidget(self._freq_spin)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        for label, delta in [("−1M", -1e6), ("−100k", -100e3),
                              ("+100k", 100e3), ("+1M", 1e6)]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, d=delta: self._step_center_freq(d))
            row2.addWidget(btn)
        gl.addLayout(row2)
        vbox.addWidget(grp)

        # ── Sample Rate ──────────────────────────────────────────────────────
        grp = _group("Sample Rate")
        gl = QVBoxLayout(grp)
        self._sr_combo = QComboBox()
        for label, _ in SAMPLE_RATES:
            self._sr_combo.addItem(label)
        self._sr_combo.setCurrentIndex(2)   # 2 MHz default
        gl.addWidget(self._sr_combo)
        vbox.addWidget(grp)

        # ── Receive Channel ──────────────────────────────────────────────────
        grp = _group("Receive Channel")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(_label("Offset:"))
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-self._sdr.sample_rate / 2 / 1e3,
                                    self._sdr.sample_rate / 2 / 1e3)
        self._offset_spin.setDecimals(1)
        self._offset_spin.setSuffix(" kHz")
        self._offset_spin.setValue(0.0)
        self._offset_spin.setSingleStep(1.0)
        row.addWidget(self._offset_spin)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(_label("Mode:"))
        self._mode_combo = QComboBox()
        for m in DEMOD_MODES:
            self._mode_combo.addItem(m.upper())
        gl.addLayout(row2)
        gl.addWidget(self._mode_combo)

        row3 = QHBoxLayout()
        row3.addWidget(_label("BW:"))
        self._bw_spin = QDoubleSpinBox()
        self._bw_spin.setRange(1.0, 500.0)
        self._bw_spin.setDecimals(1)
        self._bw_spin.setSuffix(" kHz")
        self._bw_spin.setValue(BW_PRESETS['nfm'] / 1e3)
        self._bw_spin.setSingleStep(1.0)
        row3.addWidget(self._bw_spin)
        gl.addLayout(row3)
        vbox.addWidget(grp)

        # ── Gain ─────────────────────────────────────────────────────────────
        grp = _group("Gain")
        gl = QVBoxLayout(grp)

        self._agc_check = QCheckBox("AGC (slow attack)")
        gl.addWidget(self._agc_check)

        row = QHBoxLayout()
        row.addWidget(_label("Gain:"))
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.0, 74.5)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setValue(self._sdr.gain)
        self._gain_spin.setSingleStep(1.0)
        row.addWidget(self._gain_spin)
        gl.addLayout(row)

        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_slider.setRange(0, 745)
        self._gain_slider.setValue(int(self._sdr.gain * 10))
        gl.addWidget(self._gain_slider)
        vbox.addWidget(grp)

        # ── Display ───────────────────────────────────────────────────────────
        grp = _group("Display")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(_label("Min dB:"))
        self._min_db = QSpinBox()
        self._min_db.setRange(-140, -20)
        self._min_db.setValue(-110)
        self._min_db.setSuffix(" dB")
        row.addWidget(self._min_db)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(_label("Max dB:"))
        self._max_db = QSpinBox()
        self._max_db.setRange(-100, 0)
        self._max_db.setValue(-10)
        self._max_db.setSuffix(" dB")
        row2.addWidget(self._max_db)
        gl.addLayout(row2)
        vbox.addWidget(grp)

        # ── Audio ─────────────────────────────────────────────────────────────
        grp = _group("Audio")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(_label("Vol:"))
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        row.addWidget(self._vol_slider)
        self._vol_label = QLabel("80%")
        self._vol_label.setFixedWidth(32)
        row.addWidget(self._vol_label)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(_label("SQL:"))
        self._sql_spin = QSpinBox()
        self._sql_spin.setRange(-120, 0)
        self._sql_spin.setValue(-60)
        self._sql_spin.setSuffix(" dB")
        row2.addWidget(self._sql_spin)
        gl.addLayout(row2)

        self._mute_btn = QPushButton("Mute")
        self._mute_btn.setCheckable(True)
        gl.addWidget(self._mute_btn)
        vbox.addWidget(grp)

        vbox.addStretch()
        return container

    def _build_bottom_bar(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        tx_grp = _group("TX Text")
        tx_layout = QVBoxLayout(tx_grp)
        self._tx_edit = QPlainTextEdit()
        self._tx_edit.setPlaceholderText("Enter text to send (max 1024 bytes)")
        tx_layout.addWidget(self._tx_edit)

        tx_row = QHBoxLayout()
        self._send_btn = QPushButton("Send")
        self._send_file_btn = QPushButton("Send file...")
        tx_row.addWidget(self._send_btn)
        tx_row.addWidget(self._send_file_btn)
        tx_layout.addLayout(tx_row)
        layout.addWidget(tx_grp, 1)

        rx_grp = _group("RX Text")
        rx_layout = QVBoxLayout(rx_grp)
        self._rx_display = QPlainTextEdit()
        self._rx_display.setReadOnly(True)
        self._rx_display.setPlaceholderText("Received messages appear here")
        rx_layout.addWidget(self._rx_display)
        layout.addWidget(rx_grp, 1)

        return container

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View
        view_menu = mb.addMenu("&View")
        for label, delta in [("−1 MHz", -1e6), ("−100 kHz", -100e3),
                               ("+100 kHz", 100e3), ("+1 MHz", 1e6)]:
            act = QAction(label, self)
            act.triggered.connect(lambda _, d=delta: self._step_center_freq(d))
            view_menu.addAction(act)

        # Help
        help_menu = mb.addMenu("&Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ── signal connections ───────────────────────────────────────────────────

    def _connect_signals(self):
        # SDR worker → display
        self._sdr.fft_ready.connect(self._on_fft)
        self._sdr.audio_ready.connect(self._audio.enqueue)
        self._sdr.decoded_data.connect(self._on_decoded_data)
        self._sdr.status_msg.connect(self._on_status)
        self._sdr.error_msg.connect(self._on_error)

        # Audio worker errors
        self._audio.error_msg.connect(self._on_error)

        # Display click → tune
        self._display.freq_clicked.connect(self._on_freq_clicked)

        # Controls → SDR
        self._conn_btn.clicked.connect(self._on_connect_clicked)
        self._freq_spin.valueChanged.connect(self._on_freq_spin_changed)
        self._sr_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        self._offset_spin.valueChanged.connect(self._on_demod_params_changed)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._bw_spin.valueChanged.connect(self._on_demod_params_changed)
        self._agc_check.toggled.connect(self._on_agc_toggled)
        self._gain_spin.valueChanged.connect(self._on_gain_spin_changed)
        self._gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self._min_db.valueChanged.connect(self._on_db_range_changed)
        self._max_db.valueChanged.connect(self._on_db_range_changed)
        self._vol_slider.valueChanged.connect(self._on_volume_changed)
        self._sql_spin.valueChanged.connect(
            lambda v: self._audio.set_squelch(float(v)))
        self._mute_btn.toggled.connect(self._audio.set_muted)

        self._send_btn.clicked.connect(self._on_send_clicked)
        self._send_file_btn.clicked.connect(self._on_send_file_clicked)

    # ── slot handlers ────────────────────────────────────────────────────────

    def _on_fft(self, freqs: np.ndarray, power_db: np.ndarray):
        self._display.update_fft(freqs, power_db)

    def _on_freq_clicked(self, freq_hz: float):
        """User clicked on spectrum/waterfall → set receive channel."""
        offset = freq_hz - self._center_freq
        sr_half = self._sdr.sample_rate / 2
        offset = max(-sr_half, min(sr_half, offset))
        self._rx_offset = offset
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(offset / 1e3)
        self._offset_spin.blockSignals(False)
        self._display.set_rx_offset(offset)
        self._push_demod_params()

    def _on_connect_clicked(self, checked: bool):
        if checked:
            uri = self._uri_edit.text().strip()
            ok  = self._sdr.try_connect(uri)
            self._conn_btn.setText("Disconnect" if ok else "Retry")
            self._sim_label.setVisible(self._sdr.simulation_mode)
        else:
            self._conn_btn.setText("Connect")

    def _on_freq_spin_changed(self, mhz: float):
        freq = int(mhz * 1e6)
        self._center_freq = freq
        self._sdr.set_center_freq(freq)
        self._display.set_center_freq(float(freq))

    def _step_center_freq(self, delta_hz: float):
        new_mhz = self._freq_spin.value() + delta_hz / 1e6
        new_mhz = max(70.0, min(6000.0, new_mhz))
        self._freq_spin.setValue(new_mhz)   # triggers _on_freq_spin_changed

    def _on_sample_rate_changed(self, idx: int):
        _, rate = SAMPLE_RATES[idx]
        self._sdr.set_sample_rate(rate)
        self._display.set_sample_rate(float(rate))
        half = rate / 2 / 1e3
        self._offset_spin.setRange(-half, half)

    def _on_mode_changed(self, idx: int):
        self._demod_mode = DEMOD_MODES[idx]
        # Auto-set bandwidth
        self._bw_spin.blockSignals(True)
        self._bw_spin.setValue(BW_PRESETS[self._demod_mode] / 1e3)
        self._bw_spin.blockSignals(False)
        self._push_demod_params()

    def _on_demod_params_changed(self):
        self._rx_offset = self._offset_spin.value() * 1e3
        self._display.set_rx_offset(self._rx_offset)
        self._push_demod_params()

    def _push_demod_params(self):
        self._sdr.set_demod(
            self._demod_mode,
            self._rx_offset,
            self._bw_spin.value() * 1e3,
        )

    def _on_agc_toggled(self, enabled: bool):
        mode = "slow_attack" if enabled else "manual"
        self._gain_spin.setEnabled(not enabled)
        self._gain_slider.setEnabled(not enabled)
        self._sdr.set_gain(self._gain_spin.value(), mode)

    def _on_gain_spin_changed(self, val: float):
        self._gain_slider.blockSignals(True)
        self._gain_slider.setValue(int(val * 10))
        self._gain_slider.blockSignals(False)
        self._sdr.set_gain(val, self._sdr.gain_mode)

    def _on_gain_slider_changed(self, raw: int):
        val = raw / 10.0
        self._gain_spin.blockSignals(True)
        self._gain_spin.setValue(val)
        self._gain_spin.blockSignals(False)
        self._sdr.set_gain(val, self._sdr.gain_mode)

    def _on_db_range_changed(self):
        lo = float(self._min_db.value())
        hi = float(self._max_db.value())
        if lo >= hi:
            return
        self._display.set_power_range(lo, hi)

    def _on_volume_changed(self, val: int):
        self._vol_label.setText(f"{val}%")
        self._audio.set_volume(val / 100.0)

    def _on_status(self, msg: str):
        self._sb_left.setText(msg)
        self._sim_label.setVisible(self._sdr.simulation_mode)
        if self._sdr.simulation_mode:
            self._sim_label.setText("● Simulation mode")

    def _on_error(self, msg: str):
        self._sb.showMessage(f"Error: {msg}", 3000)

    def _on_send_clicked(self):
        text = self._tx_edit.toPlainText().strip()
        if not text:
            self._on_status("Enter text before sending")
            return
        try:
            self._sdr.send_text(text)
        except Exception as exc:
            self._on_error(str(exc))

    def _on_send_file_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file to send")
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self._sdr.send_data(data)
        except Exception as exc:
            self._on_error(str(exc))

    def _on_decoded_data(self, data: bytes):
        try:
            text = data.decode('utf-8')
        except Exception:
            text = data.hex()
        self._rx_display.appendPlainText(text)
        self._on_status(f"Received {len(data)} bytes")

    def _refresh_status(self):
        mode_str = "SIM" if self._sdr.simulation_mode else "HW"
        self._sb_right.setText(
            f"{mode_str} | "
            f"CF {self._center_freq/1e6:.4f} MHz | "
            f"SR {self._sdr.sample_rate/1e6:.1f} MSPS | "
            f"{self._demod_mode.upper()} | "
            f"RX {self._rx_offset/1e3:+.1f} kHz"
        )

    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "About PySDR Receiver",
            "<b>PySDR Receiver</b><br>"
            "PlutoSDR-based software-defined radio receiver.<br><br>"
            "Built with pyadi-iio, PyQt5, pyqtgraph, and scipy.<br>"
            "Click the spectrum or waterfall to tune the receive channel.<br><br>"
            "<i>Simulation mode active when no PlutoSDR is connected.</i>"
        )

    def closeEvent(self, event):
        self._sdr.stop()
        self._audio.stop()
        event.accept()
