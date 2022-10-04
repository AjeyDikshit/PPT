# This Python file uses the following encoding: utf-8

import sys
import os
from PyQt5 import QtWidgets
# from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt6 import QtWebEngineWidgets
from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
from functions import *
import pyqtgraph as pg
import random
import math
import pandas as pd
from conversion_functions import *
from form import Ui_MainWindow as Ui_M
from convertfile import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_M):
    def __init__(self):
        # super(MainWindow, self).__init__()
        # uic.loadUi('form.ui', self)
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        self.setWindowTitle('Advanced Postprocessing')

        # Loading data in function combo box
        self.function_box.addItems(
            ['', '1. Low pass filter', '2. High pass filter', '3. Differentiation', '4. Integration',
             '5. Windowed Phasor (Magnitude)', '6. Windowed Phasor (Angle)', '7. Trend filter',
             '8. Moving window average', '9. Moving window RMS',
             '10. Clarke\'s Transform', '11. Clarke\'s Transform (Inverse)',
             '12. Park\'s Transform', '13. Park\'s Transform (Inverse)',
             '14. Sequence Transform', '15. Instantaneous LL RMS voltage', '16. Instantaneous line current'])

        # Plotting of signals from users
        self.plot_button.clicked.connect(self.plotter)

        # Selecting the function to apply
        self.function_box.activated.connect(self.selected)

        # Selecting default test case
        self.use_test.clicked.connect(self.useTest)

        # Night mode
        self.plotwidget.setBackground('w')
        self.background.clicked.connect(self.setBackG)

        # Show grid
        self.grid_check.clicked.connect(self.changeGrid)

        # Clear plot
        self.clear_button.clicked.connect(self.clearPlot)

        # Select between file input and function input
        self.fileinput.clicked.connect(self.changeFormat)

        # Default values of the buttons
        self.time_signal.setEnabled(True)
        self.signal.setEnabled(True)
        self.function_box.setEnabled(True)
        self.plot_button.setEnabled(True)
        self.file_1.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.file_signal_1.setEnabled(False)
        self.file_signal_2.setEnabled(False)
        self.file_signal_3.setEnabled(False)
        self.file_signal_4.setEnabled(False)
        self.param1.setEnabled(False)
        self.param2.setEnabled(False)
        self.param3.setEnabled(False)
        self.clear_file.setEnabled(False)

        # Browse File
        self.browse_button.clicked.connect(self.getfile)

        # Clear file input
        self.clear_file.clicked.connect(self.clearFile)

        # Help guide
        self.help_button.clicked.connect(self.guide_show)

        # About
        self.about_button.clicked.connect(self.About)

        # Get folder
        #        self.browse_file_conversion.clicked.connect(self.getfolder)

        # Conversion button
        #        self.convert_files.clicked.connect(self.convertFiles)
        self.convert_button.clicked.connect(self.openwidget)

    # ---------------------------------------------------------------------------------------------------------------

    def plotter(self):
        pen = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)

        if not (self.fileinput.isChecked()):
            if not (self.keep_plot.isChecked()):
                self.plotwidget.clear()

            if self.use_test.isChecked():
                t = np.arange(0, 1, 1e-5)
                x = np.zeros(len(t))
                for i in range(len(t)):
                    if t[i] < 0.35:
                        x[i] = 10 * np.sin(2 * np.pi * 50 * t[i])
                    elif 0.35 <= t[i] < 0.7:
                        x[i] = 10 * np.sin(2 * np.pi * 50 * t[i]) + 2 * np.sin(2 * np.pi * 500 * t[i])
                    else:
                        x[i] = 8
            else:
                t = eval(self.time_signal.text())
                x = eval(self.signal.text())

            if self.function_box.currentText() == '':
                self.plotwidget.plot(t, x, pen=pen)

            elif self.function_box.currentText()[3:] == 'Low pass filter':
                self.param1_label.setText('Enter cutoff frequency:')
                self.param1.setEnabled(True)
                y = mylowpass(t, x, float(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'High pass filter':
                self.param1_label.setText('Enter cutoff frequency:')
                self.param1.setEnabled(True)
                y = myhighpass(t, x, float(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Differentiation':
                y = derivative(t, x)
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Integration':
                y = integration(t, x)
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Windowed Phasor (Magnitude)':
                y, t_new = window_phasor_mag(t, x, int(self.param1.text()), float(self.param2.text()),
                                             int(self.param3.text()))
                self.plotwidget.plot(t_new, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Windowed Phasor (Angle)':
                y, t_new = window_phasor_angle(t, x, int(self.param1.text()), float(self.param2.text()),
                                               int(self.param3.text()))
                self.plotwidget.plot(t_new, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Trend filter':
                y = trendfilter(t, x, eval(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Moving window average':
                y = avgMovWin(t, x, int(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Moving window RMS':
                y = rmsMovWin(t, x, int(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

        else:
            if not (self.keep_plot.isChecked()):
                self.plotwidget.clear()

            df = pd.read_csv(self.file_1.text())

            t = df[self.file_signal_1.currentText()]
            x = df[self.file_signal_2.currentText()]

            if self.function_box.currentText() == '':
                self.plotwidget.plot(df[self.file_signal_1.currentText()], df[self.file_signal_2.currentText()],
                                     pen=pen)

            elif self.function_box.currentText()[3:] == 'Low pass filter':
                self.param1.setEnabled(True)
                y = mylowpass(t, x, float(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'High pass filter':
                self.param1.setEnabled(True)
                y = myhighpass(t, x, float(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Differentiation':
                y = derivative(t, x)
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Integration':
                y = integration(t, x)
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Windowed Phasor (Magnitude)':
                y, t_new = window_phasor_mag(t, x, int(self.param1.text()), float(self.param2.text()),
                                             int(self.param3.text()))
                self.plotwidget.plot(t_new, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Windowed Phasor (Angle)':
                y, t_new = window_phasor_angle(t, x, int(self.param1.text()), float(self.param2.text()),
                                               int(self.param3.text()))
                self.plotwidget.plot(t_new, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Trend filter':
                y = trendfilter(t, x, eval(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[4:] == 'Clarke\'s Transform':
                x1, y1, z1 = clarkestranform(t, x, df[self.file_signal_3.currentText()],
                                             df[self.file_signal_4.currentText()])
                pen1 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                pen2 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                self.plotwidget.plot(t, x1, pen=pen)
                self.plotwidget.plot(t, y1, pen=pen1)
                self.plotwidget.plot(t, z1, pen=pen2)

            elif self.function_box.currentText()[4:] == 'Clarke\'s Transform (Inverse)':
                x1, y1, z1 = inv_clarkestransform(t, x, df[self.file_signal_3.currentText()],
                                                  df[self.file_signal_4.currentText()])
                pen1 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                pen2 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                self.plotwidget.plot(t, x1, pen=pen)
                self.plotwidget.plot(t, y1, pen=pen1)
                self.plotwidget.plot(t, z1, pen=pen2)

            elif self.function_box.currentText()[4:] == 'Sequence Transform':
                x1, y1, z1 = sequencetransform(t, x, df[self.file_signal_3.currentText()],
                                               df[self.file_signal_4.currentText()])
                pen1 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                pen2 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                self.plotwidget.plot(t, x1, pen=pen)
                self.plotwidget.plot(t, y1, pen=pen1)
                self.plotwidget.plot(t, z1, pen=pen2)
                # QMessageBox.information(self, '', 'Complete')

            elif self.function_box.currentText()[4:] == 'Park\'s Transform':
                x1, y1, z1 = parkstransform(t, x, df[self.file_signal_3.currentText()],
                                            df[self.file_signal_4.currentText()],
                                            int(self.param2.text()), int(self.param1.text()))
                pen1 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                pen2 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                self.plotwidget.plot(t, x1, pen=pen)
                self.plotwidget.plot(t, y1, pen=pen1)
                self.plotwidget.plot(t, z1, pen=pen2)
                # QMessageBox.information(self, '', 'Complete')

            elif self.function_box.currentText()[4:] == 'Park\'s Transform (Inverse)':
                x1, y1, z1 = inv_parkstransform(t, x, df[self.file_signal_3.currentText()],
                                                df[self.file_signal_4.currentText()],
                                                int(self.param2.text()), int(self.param1.text()))
                pen1 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                pen2 = pg.mkPen(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
                self.plotwidget.plot(t, x1, pen=pen)
                self.plotwidget.plot(t, y1, pen=pen1)
                self.plotwidget.plot(t, z1, pen=pen2)

            elif self.function_box.currentText()[3:] == 'Moving window average':
                y = avgMovWin(t, x, int(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[3:] == 'Moving window RMS':
                y = rmsMovWin(t, x, int(self.param1.text()))
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[4:] == 'Instantaneous LL RMS voltage':
                y = instaLL_RMSVoltage(t, x, df[self.file_signal_3.currentText()], df[self.file_signal_4.currentText()])
                self.plotwidget.plot(t, y, pen=pen)

            elif self.function_box.currentText()[4:] == 'Instantaneous line current':
                y = insta_RMSCurrent(t, x, df[self.file_signal_3.currentText()], df[self.file_signal_4.currentText()])
                self.plotwidget.plot(t, y, pen=pen)

    # ----------------------------------------------------------------------------------------

    def selected(self):
        self.file_signal_3.setEnabled(False)
        self.file_signal_4.setEnabled(False)
        self.param1.clear()
        self.param2.clear()
        self.param3.clear()
        self.param1_label.clear()
        self.param2_label.clear()
        self.param3_label.clear()
        self.param1.setEnabled(False)
        self.param2.setEnabled(False)
        self.param3.setEnabled(False)

        if self.function_box.currentText()[3:] in ['Low pass filter', 'High pass filter']:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1_label.setText('Enter cutoff frequency:')
            self.param1.setEnabled(True)

        elif self.function_box.currentText()[3:] in ['Windowed Phasor (Magnitude)', 'Windowed Phasor (Angle)']:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1_label.setText('Factor to down-sample')
            self.param2_label.setText('Cycles to average')
            self.param3_label.setText('Enter dominant frequency')
            self.param1.setEnabled(True)
            self.param2.setEnabled(True)
            self.param3.setEnabled(True)

        elif self.function_box.currentText()[3:] in ['Integration', 'Differentiation']:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1_label.setText('No parameter required')

        elif self.function_box.currentText()[3:] == 'Trend filter':
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1_label.setText('Enter lambda value')
            self.param1.setEnabled(True)

        elif self.function_box.currentText()[4:] == 'Clarke\'s Transform':
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('No parameter required')

        elif self.function_box.currentText()[4:] == 'Clarke\'s Transform (Inverse)':
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('No parameter required')

        elif self.function_box.currentText()[4:] == 'Sequence Transform':
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('No parameter required')

        elif self.function_box.currentText()[4:] == 'Park\'s Transform':
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('Gamma:')
            self.param2_label.setText('Omega (frequency):')
            self.param1.setEnabled(True)
            self.param2.setEnabled(True)

        elif self.function_box.currentText()[4:] == 'Park\'s Transform (Inverse)':
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('Gamma:')
            self.param2_label.setText('Omega (frequency):')
            self.param1.setEnabled(True)
            self.param2.setEnabled(True)

        elif self.function_box.currentText()[3:] in ['Moving window average', 'Moving window RMS']:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1_label.setText('Enter time window')
            self.param1.setEnabled(True)

        elif self.function_box.currentText()[4:] in ['Instantaneous LL RMS voltage', 'Instantaneous line current']:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.file_signal_3.setEnabled(True)
            self.file_signal_4.setEnabled(True)
            self.param1_label.setText('No parameters required')

        else:
            self.file_signal_3.setCurrentIndex(0)
            self.file_signal_4.setCurrentIndex(0)
            self.param1.setEnabled(False)
            self.param2.setEnabled(False)
            self.param3.setEnabled(False)

    # --------------------------------------------------------------------------------------------

    def clearPlot(self):
        self.plotwidget.clear()

    def changeGrid(self):
        if self.grid_check.isChecked():
            self.plotwidget.showGrid(x=True, y=True, alpha=1)
        else:
            self.plotwidget.showGrid(x=False, y=False)

    def setBackG(self):
        if self.background.isChecked():
            self.plotwidget.setBackground('black')
        else:
            self.plotwidget.setBackground('w')

    def useTest(self):
        if self.use_test.isChecked():
            self.time_signal.clear()
            self.signal.clear()
            self.time_signal.setEnabled(False)
            self.signal.setEnabled(False)
        else:
            self.time_signal.setEnabled(True)
            self.signal.setEnabled(True)

    # --------------------------------------------------------------------------------------------------------

    def changeFormat(self):
        if self.fileinput.isChecked():
            if self.use_test.isChecked():
                self.use_test.click()
            self.time_signal.clear()
            self.signal.clear()
            self.time_signal.setEnabled(False)
            self.signal.setEnabled(False)
            self.function_box.setEnabled(True)
            self.plot_button.setEnabled(True)
            #            self.select_button.setEnabled(True)
            self.file_1.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.file_signal_1.setEnabled(True)
            self.file_signal_2.setEnabled(True)
            self.use_test.setEnabled(False)
            self.clear_file.setEnabled(True)
        else:
            self.file_signal_1.clear()
            self.file_signal_2.clear()
            self.time_signal.clear()
            self.signal.clear()
            self.file_1.clear()
            self.time_signal.setEnabled(True)
            self.signal.setEnabled(True)
            self.function_box.setEnabled(True)
            self.plot_button.setEnabled(True)
            #            self.select_button.setEnabled(True)
            self.file_1.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.file_signal_1.setEnabled(False)
            self.file_signal_2.setEnabled(False)
            self.use_test.setEnabled(True)
            self.clear_file.setEnabled(False)

    # ---------------------------------------------------------------------------------------------

    def getfile(self):
        self.file_signal_1.clear()
        self.file_signal_2.clear()
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode
        filenames = QStringListModel()

        if dlg.exec():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            with f:
                data = f.read()
                self.file_1.setText(filenames[0])

        df = pd.read_csv(self.file_1.text())

        self.file_signal_1.clear()
        self.file_signal_2.clear()
        self.file_signal_3.clear()
        self.file_signal_4.clear()

        self.file_signal_1.addItems([''])
        self.file_signal_1.addItems(df.columns)
        self.file_signal_2.addItems([''])
        self.file_signal_2.addItems(df.columns)
        self.file_signal_3.addItems([''])
        self.file_signal_3.addItems(df.columns)
        self.file_signal_4.addItems([''])
        self.file_signal_4.addItems(df.columns)

    # -----------------------------------------------------------------------------------------------------

    def clearFile(self):
        self.file_1.clear()
        self.file_signal_1.clear()
        self.file_signal_2.clear()

    def About(self):
        QtWidgets.QMessageBox.information(self, 'About', 'Created by Ajey Dikshit \n   September 2022')

    # -----------------------------------------------------------------------------------------------------

    def guide_show(self):
        self.guide = guide1()
        self.guide.show()

    # -----------------------------------------------------------------------------------------------------

    def openwidget(self):
        self.widget = convertFiles1()
        self.widget.show()


# -----------------------------------------------------------------------------------------------------

class convertFiles1(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # super(convertFiles1, self).__init__()
        # uic.loadUi('convertfile.ui', self)
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        self.setWindowTitle('Convert files')

        self.browse_folder.clicked.connect(self.getfolder)
        self.convert_files.clicked.connect(self.convertFiles)

    def getfolder(self):
        dlg = QtWidgets.QFileDialog(self)
        self.dir_path = dlg.getExistingDirectory(self, 'Choose directory', 'C:\\')
        self.folder_location.setText(self.dir_path)

    def convertFiles(self):
        path = self.folder_location.text()

        com_files, mat_files, inf_files, out_files = files2convert(path)

        for i in range(len(com_files)):
            cfg_file = com_files[i]
            dat_file = com_files[i][:-4] + '.dat'
            comtrade2csv(cfg_file, dat_file, path)

        for i in mat_files:
            mat2csv(i, path)

        for i in inf_files:
            pscad2csv(i, out_files, path)

        print('All files converted!')
        QtWidgets.QMessageBox.information(self, 'Complete', 'All files converted!')


# ------------------------------------------------------------------------------------------------------

class guide1(QtWebEngineWidgets.QWebEngineView):
    def __init__(self):
        super(guide1, self).__init__()
        self.setWindowTitle('Tutorial')
        self.load(QUrl('https://drive.google.com/file/d/1zM1OHlAPwDCblt4KhIw_cARTkU13apvj/view?usp=sharing'))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
