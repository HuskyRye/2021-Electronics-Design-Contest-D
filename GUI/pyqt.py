import sys
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWebChannel import *
import numpy as np
from numpy import *
import cv2
import math
from timeit import default_timer as timer
import time
import os
# import wiringpi
# wiringpi.piHiPri(1)


class Backend(QObject):
    dataUpdateSignal = pyqtSignal(str, str)
    videoUpdateSignal = pyqtSignal(object, object)
    gravityUpdateSignal = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.show = True
        self.start = False
        self.calibrating = False
        self.center = False
        self.degrees = []
        self.alpha_times = []
        self.alpha_lasttime = None
        self.beta_times = []
        self.beta_lasttime = None

    def onCapture(self, frame1, frame2):
        frame1, alpha, gama = self.detect(frame1)
        frame2, beta, delta = self.detect(frame2)
        if self.start:
            x0, y0, zA, zB = self.getXYZ(alpha, gama, beta, delta, 65.8, 40)
            degree = math.atan(y0/x0) * 180 / math.pi if x0 != 0 else 90
            self.degrees.append(degree)
            thistime = timer()
            if alpha < 0.1:
                if not self.alpha_lasttime:
                    self.alpha_times.append(thistime)
                    self.alpha_lasttime = thistime
                if self.alpha_lasttime and thistime - self.alpha_lasttime > 0.5:
                    self.alpha_times.append(thistime)
                    self.alpha_lasttime = thistime
            if beta < 0.1:
                if not self.beta_lasttime:
                    self.beta_times.append(thistime)
                    self.beta_lasttime = thistime
                if self.beta_lasttime and thistime - self.beta_lasttime > 0.5:
                    self.beta_times.append(thistime)
                    self.beta_lasttime = thistime

            if len(self.degrees) >= 100:
                degree = round(abs(
                    mean(sort(self.degrees)[int(0.1*len(self.degrees)):int(0.9*len(self.degrees))])), 2)
                alpha_times = [self.alpha_times[i] - self.alpha_times[i-2]
                               for i in range(2, len(self.alpha_times))]
                alpha_period = mean(alpha_times)

                beta_times = [self.beta_times[i] - self.beta_times[i-2]
                              for i in range(2, len(self.beta_times))]
                beta_period = mean(beta_times)
                # period = mean(
                #     sort(times)[int(0.1*len(times)):int(0.9*len(times))])

                if degree < 30:
                    period = beta_period
                elif degree < 60:
                    period = (alpha_period + beta_period) / 2
                else:
                    period = alpha_period
                half_pen_length = 7
                length = round(period * period * 9.7915 /
                               4 / math.pi / math.pi * 100 - half_pen_length, 1)
                self.dataUpdateSignal.emit(str(length), str(degree))
                self.start = False
                self.degrees = []
                self.alpha_times = []
                self.alpha_lasttime = None
                self.beta_times = []
                self.beta_lasttime = None
                QApplication.beep()
                os.system("aplay sound.wav")

        if self.calibrating:
            time.sleep(5)
            self.gravityUpdateSignal.emit('9.791')
            self.calibrating = False
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        qimg1 = QImage(frame1.data, frame1.shape[1], frame1.shape[0],
                       QImage.Format_RGB888)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        qimg2 = QImage(frame2.data, frame2.shape[1], frame2.shape[0],
                       QImage.Format_RGB888)
        self.videoUpdateSignal.emit(qimg1, qimg2)

    def getXYZ(self, Aalpha, Agama, Bbeta, Bdelta, phiHor, phiVer):
        phiHor = (phiHor/2)*math.pi/180
        phiVer = (phiVer/2)*math.pi/180
        x = (Aalpha * Bbeta * (math.tan(phiHor)**2) - Bbeta *
             math.tan(phiHor)) / (Aalpha * Bbeta * (math.tan(phiHor)**2) - 1)
        y = (Aalpha * Bbeta * (math.tan(phiHor)**2) - Aalpha *
             math.tan(phiHor)) / (Aalpha * Bbeta * (math.tan(phiHor)**2) - 1)
        zA = Agama * math.tan(phiVer) * (1 - x)
        zB = Bdelta * math.tan(phiVer) * (1 - y)
        return x, y, zA, zB

    def detect(self, frame):
        colorSplit = cv2.split(frame)
        redObject = cv2.subtract(colorSplit[2], colorSplit[1])
        ret, redObject = cv2.threshold(redObject, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        redObject = cv2.morphologyEx(redObject, cv2.MORPH_OPEN, kernel)
        cnts = cv2.findContours(redObject, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        del redObject
        possible_pen = list()
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            if max(rect[1]) < 4:
                continue
            if max(rect[1]) < 3 * min(rect[1]):
                continue
            box = cv2.boxPoints(rect)
            bounding = cv2.boundingRect(box)
            if bounding[2] * 1 > bounding[3]:
                continue
            possible_pen.append(box)
        if len(possible_pen):
            possible_pen.sort(
                key=lambda pen: cv2.contourArea(pen), reverse=False)
            pen = possible_pen[0]
            centerx = sum(pen[:, 0]) / 4
            centery = sum(pen[:, 1]) / 4
            frameSize = frame.shape
            alpha = abs(centerx-frameSize[1]/2) / (frameSize[1]/2)
            gama = abs(centery-frameSize[0]/2) / (frameSize[0]/2)
            if self.show:
                cv2.drawContours(frame, [np.int0(pen)], -1, (0, 0, 255), 3)
                cv2.circle(frame, (int(centerx), int(centery)),
                           3, (0, 255, 0), -1)
            if self.center:
                cv2.putText(frame, "a={}".format(round(alpha, 4)),
                            (75, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, "b={}".format(round(gama, 4)),
                            (75, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6, cv2.LINE_AA)
            return frame, alpha, gama
        else:
            return frame, -1, -1


class MultiVideoProcess(QObject):
    captrueFrameSignal = pyqtSignal(object, object)

    def __init__(self, capture1, capture2):
        super(QObject, self).__init__()
        self.capture1 = capture1
        self.capture2 = capture2

    def run(self):
        while True:
            self.cap1 = cv2.VideoCapture(self.capture1)
            self.cap2 = cv2.VideoCapture(self.capture2)
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if not ret1 or not ret2:
                continue
            frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
            frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
            self.captrueFrameSignal.emit(frame1, frame2)
            self.cap1.release()
            self.cap2.release()


class Handle(QObject):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @ pyqtSlot()
    def on_start(self):
        self.backend.start = True

    @ pyqtSlot()
    def quit(self):
        app.quit()

    @ pyqtSlot()
    def toggleshow(self):
        self.backend.show = not self.backend.show

    @ pyqtSlot(float)
    def set_L(self, L):
        self.backend.L = L

    @ pyqtSlot()
    def togglecenter(self):
        self.backend.center = not self.backend.center

    @ pyqtSlot(float)
    def calibrate(self, length):
        self.backend.calibrating = True


class VideoLabel(QLabel):
    def __init__(self, parent):
        super(VideoLabel, self).__init__(parent)
        self.isFullScreen = False
        self.setScaledContents(True)
        # self.animation.setEasingCurve(QEasingCurve.InOutQuad)

    def mouseDoubleClickEvent(self, event):
        if not self.isFullScreen:
            self.g = self.geometry()
            win.browser.hide()
            win.video_label1.hide()
            win.video_label2.hide()
            self.show()
            self.raise_()
            self.isFullScreen = True
            self.animation = QPropertyAnimation(self, b"geometry")
            self.animation.setDuration(500)
            self.animation.setEndValue(QRect(656, 0, 608, 1080))
            self.animation.finished.connect(self.animationFinished)
            win.setStyleSheet("background-color: black")
            self.animation.start()
        else:
            self.animation = QPropertyAnimation(self, b"geometry")
            self.animation.setDuration(500)
            self.animation.setEndValue(self.g)
            self.animation.finished.connect(self.animationFinished)
            self.animation.start()
            win.browser.show()
            win.video_label1.show()
            win.video_label2.show()
            win.setStyleSheet("background-color: white")

            self.isFullScreen = False

    def animationFinished(self):
        self.animation = None

    def paintEvent(self, event):
        super(VideoLabel, self).paintEvent(event)
        if not self.isFullScreen:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 255, 255), 32))
            painter.drawRoundedRect(
                self.rect(), 38, 38)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('全国大学生电子设计竞赛')
        desktop = QApplication.desktop()
        self.setGeometry(desktop.screenGeometry(
            0 if desktop.screenCount() == 1 else 1))
        self.browser = QWebEngineView(self)
        self.browser.load(
            QUrl('file:///'+QFileInfo('gui2.html').absoluteFilePath()))
        self.setCentralWidget(self.browser)
        # self.setStyleSheet("background-color: black")
        self.backend = Backend()
        self.backend.dataUpdateSignal.connect(self.updateData)
        self.backend.videoUpdateSignal.connect(self.updateVideo)
        self.backend.gravityUpdateSignal.connect(self.updateGravity)
        self.backend_thread = QThread()
        self.backend.moveToThread(self.backend_thread)
        # self.backend_thread.started.connect(self.backend.run)

        self.channel = QWebChannel()
        self.handel = Handle(self.backend)
        self.channel.registerObject('obj', self.handel)
        self.browser.page().setWebChannel(self.channel)

        self.video_label1 = VideoLabel(self)
        self.video_label1.setGeometry(13, 22, 585, 1040)
        self.video_label2 = VideoLabel(self)
        self.video_label2.setGeometry(605, 22, 585, 1040)

        self.multiVideoProcess = MultiVideoProcess(
            'http://169.254.18.129:8080/?action=stream',
            'http://169.254.47.143:8080/?action=stream',)
        self.multiVideoProcess.captrueFrameSignal.connect(
            self.backend.onCapture)
        self.multi_video_thread = QThread()
        self.multiVideoProcess.moveToThread(self.multi_video_thread)
        self.multi_video_thread.started.connect(self.multiVideoProcess.run)

        self.backend_thread.start()
        self.multi_video_thread.start()

    def updateData(self, length, degree):
        self.browser.page().runJavaScript('updateData("{}", "{}")'.format(length, degree))

    def updateGravity(self, gravity):
        self.browser.page().runJavaScript('updateGravity("{}")'.format(gravity))

    def updateVideo(self, video1, video2):
        self.video_label1.setPixmap(QPixmap.fromImage(video1))
        self.video_label2.setPixmap(QPixmap.fromImage(video2))


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        win = MainWindow()
        win.showFullScreen()
        # win.show()
        app.exit(app.exec_())
    except Exception as reason:
        print(str(reason))
