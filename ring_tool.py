import logging

import plots as p
from PyQt4 import QtGui

from gui.ring import RingWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ring').setLevel(logging.INFO)
logging.getLogger('hhlab').setLevel(logging.INFO)
logging.getLogger('gui').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PyQt4').setLevel(logging.ERROR)

if __name__ == '__main__':
    import sys
    import os

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
