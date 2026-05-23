import sys
import os

# Ensure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QFont
import pyqtgraph as pg

from main_window import MainWindow


def _light_palette(app: QApplication) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(240, 240, 245))
    p.setColor(QPalette.WindowText,      QColor(30,  30,  30))
    p.setColor(QPalette.Base,            QColor(255, 255, 255))
    p.setColor(QPalette.AlternateBase,   QColor(230, 230, 235))
    p.setColor(QPalette.ToolTipBase,     QColor(255, 255, 220))
    p.setColor(QPalette.ToolTipText,     QColor(30,  30,  30))
    p.setColor(QPalette.Text,            QColor(30,  30,  30))
    p.setColor(QPalette.Button,          QColor(225, 225, 230))
    p.setColor(QPalette.ButtonText,      QColor(30,  30,  30))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Link,            QColor(0,   100, 200))
    p.setColor(QPalette.Highlight,       QColor(0,   120, 215))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(160, 160, 165))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 165))
    return p


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("PySDR Receiver")
    app.setOrganizationName("PySDR")
    app.setStyle("Fusion")
    app.setPalette(_light_palette(app))
    app.setFont(QFont("Arial", 11))

    # pyqtgraph global defaults
    pg.setConfigOptions(antialias=False, useOpenGL=False, foreground='#333', background='#f0f2f5')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
