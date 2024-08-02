# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\User\OneDrive - 國立成功大學 National Cheng Kung University\桌面\Hw2_02_E24074059_林烔_V1\HW2_2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        
        MainWindow.resize(400, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Q = QtWidgets.QGroupBox(self.centralwidget)
        self.Q.setGeometry(QtCore.QRect(20, 10, 220, 130))
        self.Q.setObjectName("Q1")
        self.q1 = QtWidgets.QPushButton(self.Q)
        self.q1.setGeometry(QtCore.QRect(20, 20, 151, 31))
        self.q1.setObjectName("q1")
        self.q2 = QtWidgets.QPushButton(self.Q)
        self.q2.setGeometry(QtCore.QRect(20, 50, 151, 31))
        self.q2.setObjectName("q2")
        self.q3 = QtWidgets.QPushButton(self.Q)
        self.q3.setGeometry(QtCore.QRect(20, 80, 151, 31))
        self.q3.setObjectName("q3")
        
        self.Q1 = QtWidgets.QGroupBox(self.centralwidget)
        self.Q1.setGeometry(QtCore.QRect(20, 150, 220, 100))
        self.Q1.setObjectName("Q1")
        self.q1_1 = QtWidgets.QPushButton(self.Q1)
        self.q1_1.setGeometry(QtCore.QRect(20, 30, 151, 31))
        self.q1_1.setObjectName("q1_1")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 250, 220, 130))
        self.groupBox.setObjectName("groupBox")
        self.q2_1 = QtWidgets.QPushButton(self.groupBox)
        self.q2_1.setGeometry(QtCore.QRect(20, 30, 151, 31))
        self.q2_1.setObjectName("q2_1")
        self.q2_2 = QtWidgets.QPushButton(self.groupBox)
        self.q2_2.setGeometry(QtCore.QRect(20, 80, 151, 31))
        self.q2_2.setObjectName("q2_2")
        self.Q3 = QtWidgets.QGroupBox(self.centralwidget)
        self.Q3.setGeometry(QtCore.QRect(20, 380, 220, 100))
        self.Q3.setObjectName("Q3")
        self.q3_1 = QtWidgets.QPushButton(self.Q3)
        self.q3_1.setGeometry(QtCore.QRect(20, 30, 170,31))
        self.q3_1.setObjectName("q3_1")
        self.Q4 = QtWidgets.QGroupBox(self.centralwidget)
        self.Q4.setGeometry(QtCore.QRect(20, 500, 220, 130))
        self.Q4.setObjectName("Q4")
        self.q4_1 = QtWidgets.QPushButton(self.Q4)
        self.q4_1.setGeometry(QtCore.QRect(20, 30, 180, 31))
        self.q4_1.setObjectName("q4_1")
        self.q4_2 = QtWidgets.QPushButton(self.Q4)
        self.q4_2.setGeometry(QtCore.QRect(20, 80, 180, 31))
        self.q4_2.setObjectName("q4_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 849, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "N26112479_HW2"))

        self.q1.setText(_translate("MainWindow", "Load video"))
        self.q2.setText(_translate("MainWindow", "Load image"))
        self.q3.setText(_translate("MainWindow", "Load Folder"))

        self.Q1.setTitle(_translate("MainWindow", "Background Subtraction "))
        self.q1_1.setText(_translate("MainWindow", "Background Subtraction "))
        
        self.groupBox.setTitle(_translate("MainWindow", "Optical Flow"))
        self.q2_1.setText(_translate("MainWindow", "Preprocessing"))
        self.q2_2.setText(_translate("MainWindow", "Video tracking"))
        
        self.Q3.setTitle(_translate("MainWindow", "Perspective Transform"))
        self.q3_1.setText(_translate("MainWindow", "Perspective Transform"))

        self.Q4.setTitle(_translate("MainWindow", "PCA"))
        self.q4_1.setText(_translate("MainWindow", "Image Reconstruction"))
        self.q4_2.setText(_translate("MainWindow", "Compute the reconstruction error"))

       
        