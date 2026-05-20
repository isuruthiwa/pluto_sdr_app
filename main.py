import sys
import os

# Ensure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QFont
import pyqtgraph as pg

from main_window import MainWindow


def _dark_palette(app: QApplication) -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(22,  22,  30))
    p.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.Base,            QColor(14,  14,  20))
    p.setColor(QPalette.AlternateBase,   QColor(35,  35,  45))
    p.setColor(QPalette.ToolTipBase,     QColor(40,  40,  50))
    p.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.Button,          QColor(45,  45,  58))
    p.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Link,            QColor(42,  130, 218))
    p.setColor(QPalette.Highlight,       QColor(42,  130, 218))
    p.setColor(QPalette.HighlightedText, Qt.black)
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(100, 100, 110))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 110))
    return p


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("PySDR Receiver")
    app.setOrganizationName("PySDR")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(app))
    app.setFont(QFont("Segoe UI", 9))

    # pyqtgraph global defaults
    pg.setConfigOptions(antialias=False, useOpenGL=False, foreground='#ccc', background='#0d0d1a')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
