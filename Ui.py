import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDialog
from PyQt5.QtCore import Qt
import pic_rc
import time
from PyQt5.QtCore import *
import os
import tensorflow as tf
from model import yolov3
from utils.data_aug import letterbox_resize
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from button_logic_mp_new  import *
import button_logic_mp_new as bl
from args import *
os.environ['CUDA_VISIBLE_DEVICES']='0'

 # 对QDialog类重写，重构closeEvent方法，实现对进度条线程的关闭
class Dialog(QtWidgets.QDialog):
    def closeEvent(self, event):
        global thread_flag
        thread_flag = False

#批量图片处理进度条窗口
class Ui_PicFolderProgress_Dialog(object):
    def setupUi(self, Dialog,file_path,out_path):
        Dialog.setObjectName("Dialog")
        Dialog.resize(504, 134)
        Dialog.setStyleSheet("QDialog\n"
"{\n"
"background:#ffffff;\n"
"}")
        self.ProgressBar = QtWidgets.QProgressBar(Dialog)
        self.ProgressBar.setGeometry(QtCore.QRect(40, 40, 431, 31))
        self.ProgressBar.setStyleSheet("QProgressBar \n"
"{   \n"
"border: 2px solid grey;   \n"
"border-radius: 5px;   \n"
"background-color: #FFFFFF;\n"
"}\n"
"\n"
"QProgressBar:chunk \n"
"{   \n"
"background-color: #8EB4E3;   \n"
"width: 20px;\n"
"}\n"
"\n"
"QProgressBar \n"
"{   \n"
"border: 2px solid #00b0f0;   \n"
"border-radius: 5px;   \n"
"text-align: center;\n"
"font: 25 15pt \"微软雅黑 Light\";\n"
"}")
        self.ProgressBar.setProperty("value", 0)
        self.ProgressBar.setObjectName("ProgressBar")
        self.StateLabel = QtWidgets.QLabel(Dialog)
        self.StateLabel.setGeometry(QtCore.QRect(10, 110, 191, 16))
        self.StateLabel.setStyleSheet("QLabel\n"
"{\n"
"font: 25 8pt \"微软雅黑 Light\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"}")
        self.StateLabel.setObjectName("StateLabel")

        # 创建并启用子线程
        self.thread_1 = PicFolderProgress_Worker(file_path,out_path)
        self.thread_1.progressBarValue.connect(self.copy_file)
        self.thread_1.start()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def copy_file(self, num):
        self.ProgressBar.setValue(num)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "图片识别进度"))
        self.StateLabel.setText(_translate("Dialog", "正在识别......"))

    #进度状态函数，默认显示“正在识别......”，输入字符串改变状态
    def StateTextChange(self,state_str):
        self.StateLabel.setText(QtCore.QCoreApplication.translate("Dialog", state_str))

#批量图片识别进度条线程
#改变进度条集进程，在run中操作   
class PicFolderProgress_Worker(QThread):
    progressBarValue = pyqtSignal(int)  # 更新进度条

    def __init__(self,file_path,out_path):
        super(PicFolderProgress_Worker, self).__init__()
        self.file_path = file_path
        self.out_path = out_path
    #这里增加了 thread_flag 判断窗口是否关闭 如果窗口已经关闭会加入else关闭循环从而结束该线程 
    def run(self):
        global thread_flag
        thread_flag = True
        index = 0
        file_path = self.file_path
        out_path = self.out_path
        if file_path == '' or out_path == '':
            # self.ProgressWindow_Close()
            return
        files = []
        for file in os.listdir(file_path):
            if file.split('.')[1] == 'jpg' or file.split('.')[1] == 'png':
                files.append(file)
        lenth = len(files)
        child_picfolder_progress_ui.StateTextChange("正在识别...    共" + str(lenth) + "张图片")
        for file in files:
            if thread_flag:
                batch_picture_op(file, file_path, out_path, args, sess, boxes, scores, labels, input_data)
                index += 1
                print(thread_flag)
                if index % 2 == 0:
                    persentage = 100 * index // lenth
                    self.progressBarValue.emit(persentage)
        self.progressBarValue.emit(100)
        child_picfolder_progress_ui.StateTextChange("识别完成")
    #进度条窗口关闭函数
    def ProgressWindow_Close(self):
        dialog_picfolder_progress.close() 

class Ui_PicFolderSelect_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(595, 308)
        Dialog.setStyleSheet("QMainWindow\n"
"{\n"
"background:#ffffff;\n"
"}")
        self.SelectToolButton = QtWidgets.QToolButton(Dialog)
        self.SelectToolButton.setGeometry(QtCore.QRect(460, 70, 50, 25))
        self.SelectToolButton.setStyleSheet("QToolButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:hover\n"
"{\n"
"background:#7BC7E9;\n"
"color:#ffffff;\n"
"font-size:16px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:pressed\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}")
        self.SelectToolButton.setObjectName("SelectToolButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(90, 30, 280, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel\n"
"{\n"
"font:14pt \"微软雅黑\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"}")
        self.label.setObjectName("label")
        self.SelectLineEdit = QtWidgets.QLineEdit(Dialog)
        self.SelectLineEdit.setGeometry(QtCore.QRect(90, 70, 360, 25))
        self.SelectLineEdit.setStyleSheet("")
        self.SelectLineEdit.setText("")
        self.SelectLineEdit.setObjectName("SelectLineEdit")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(90, 130, 280, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("QLabel\n"
"{\n"
"font:14pt \"微软雅黑\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"}")
        self.label_2.setObjectName("label_2")
        self.OutputLineEdit = QtWidgets.QLineEdit(Dialog)
        self.OutputLineEdit.setGeometry(QtCore.QRect(90, 170, 360, 25))
        self.OutputLineEdit.setText("")
        self.OutputLineEdit.setObjectName("OutputLineEdit")
        self.OutputToolButton = QtWidgets.QToolButton(Dialog)
        self.OutputToolButton.setGeometry(QtCore.QRect(460, 170, 50, 25))
        self.OutputToolButton.setStyleSheet("QToolButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:hover\n"
"{\n"
"background:#7BC7E9;\n"
"color:#ffffff;\n"
"font-size:16px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:pressed\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}")
        self.OutputToolButton.setObjectName("OutputToolButton")
        self.ConfirmButton = QtWidgets.QPushButton(Dialog)
        self.ConfirmButton.setGeometry(QtCore.QRect(120, 230, 120, 50))
        self.ConfirmButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-radius:18px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:16pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:18px;\n"
"}")
        self.ConfirmButton.setObjectName("ConfirmButton")
        self.CancelButton = QtWidgets.QPushButton(Dialog)
        self.CancelButton.setGeometry(QtCore.QRect(350, 230, 120, 50))
        self.CancelButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-radius:18px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:16pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:18px;\n"
"}")
        self.CancelButton.setObjectName("CancelButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.SelectToolButton.clicked.connect(self.Function_SelectToolButton)
        self.OutputToolButton.clicked.connect(self.Function_OutputToolButton)
        self.ConfirmButton.clicked.connect(self.Function_ConfirmButton)
        self.CancelButton.clicked.connect(self.Function_CancelButton)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "批量图片识别"))
        self.SelectToolButton.setText(_translate("Dialog", "..."))
        self.label.setText(_translate("Dialog", "选取图片存放文件夹"))
        self.label_2.setText(_translate("Dialog", "选取数据输出文件夹"))
        self.OutputToolButton.setText(_translate("Dialog", "..."))
        self.ConfirmButton.setText(_translate("Dialog", "确定"))
        self.CancelButton.setText(_translate("Dialog", "取消"))

    #选取图片文件夹按钮
    def Function_SelectToolButton(self):
        select_info = QFileDialog.getExistingDirectory(
            QtWidgets.QMainWindow(), '选择文件夹', '',)
        self.SelectFolder_ShowStr(select_info) #select_info 文件夹
        #执行函数

    #数据输出文件夹按钮
    def Function_OutputToolButton(self):
        output_info = QFileDialog.getExistingDirectory(
            QtWidgets.QMainWindow(), '选择文件夹', '',) #output_info 文件夹
        self.OutputFolder_ShowStr(output_info)
        #执行函数
    
    #确认按钮,开始识别,这个函数中的代码暂时只承担了打开下一级窗口的功能，
    #点击确认后具体要执行的任务代码请加在函数内
    def Function_ConfirmButton(self):
        self.SelectLineEdit.setText(
            QtCore.QCoreApplication.translate("Dialog",self.SelectLineEdit.text()))
        self.OutputLineEdit.setText(
            QtCore.QCoreApplication.translate("Dialog",self.OutputLineEdit.text()))
             
        global dialog_picfolder_progress       
        dialog_picfolder_progress = Dialog()
        global child_picfolder_progress_ui
        file_path = self.SelectLineEdit.text()
        out_path = self.OutputLineEdit.text()
        child_picfolder_progress_ui = Ui_PicFolderProgress_Dialog()  #进度条窗口的实例化，要改变进度条状态调用方法内函数
        child_picfolder_progress_ui.setupUi(dialog_picfolder_progress,file_path,out_path)
        dialog_picfolder_progress.setWindowModality(Qt.ApplicationModal)
        #执行函数
        dialog_picfolder_progress.show()
        dialog_picfolder_progress.exec()


    #取消按钮,取消识别,这个函数中的代码暂时只承担了关闭当前窗口的功能，
    #点击取消后具体要执行的任务代码请加在函数内
    def Function_CancelButton(self):
        #执行函数
        dialog_picfolder.close()


    #视频文件路径显示
    def SelectFolder_ShowStr(self,folderstr):
        self.SelectLineEdit.setText(QtCore.QCoreApplication.translate("Dialog", folderstr))

    #数据输出文件夹选取路径显示
    def OutputFolder_ShowStr(self,folderstr):
        self.OutputLineEdit.setText(QtCore.QCoreApplication.translate("Dialog", folderstr))

class Ui_Settings_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(681, 344)
        self.SaveRadioButton = QtWidgets.QRadioButton(Dialog)
        self.SaveRadioButton.setGeometry(QtCore.QRect(40, 90, 171, 41))
        self.SaveRadioButton.setStyleSheet("QRadioButton\n"
"{\n"
"font:14pt \"微软雅黑\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"width: 17px;\n"
"height: 17px;\n"
"spacing: 5px;\n"
"}\n"
"")
        self.SaveRadioButton.setObjectName("SaveRadioButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(220, 50, 280, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel\n"
"{\n"
"font:14pt \"微软雅黑\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"}")
        self.label.setObjectName("label")
        self.SaveLineEdit = QtWidgets.QLineEdit(Dialog)
        self.SaveLineEdit.setGeometry(QtCore.QRect(220, 100, 360, 25))
        self.SaveLineEdit.setStyleSheet("")
        self.SaveLineEdit.setText("")
        self.SaveLineEdit.setObjectName("SaveLineEdit")
        self.SaveToolButton = QtWidgets.QToolButton(Dialog)
        self.SaveToolButton.setGeometry(QtCore.QRect(590, 100, 50, 25))
        self.SaveToolButton.setStyleSheet("QToolButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:hover\n"
"{\n"
"background:#7BC7E9;\n"
"color:#ffffff;\n"
"font-size:16px;\n"
"border-radius:3px;\n"
"}\n"
"\n"
"QToolButton:pressed\n"
"{\n"
"background:#00b0f0;\n"
"color:#ffffff;\n"
"font-size:18px;\n"
"border-radius:3px;\n"
"}")
        self.SaveToolButton.setObjectName("SaveToolButton")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(80, 160, 121, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel\n"
"{\n"
"font:14pt \"微软雅黑\";\n"
"color:#0F224E;\n"
"border-radius:3px;\n"
"}")
        self.label_3.setObjectName("label_3")
        self.CameraNumLineEdit = QtWidgets.QLineEdit(Dialog)
        self.CameraNumLineEdit.setGeometry(QtCore.QRect(220, 170, 360, 25))
        self.CameraNumLineEdit.setStyleSheet("")
        self.CameraNumLineEdit.setText("")
        self.CameraNumLineEdit.setObjectName("CameraNumLineEdit")
        self.ConfirmButton = QtWidgets.QPushButton(Dialog)
        self.ConfirmButton.setGeometry(QtCore.QRect(160, 240, 120, 50))
        self.ConfirmButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-radius:18px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:16pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:18px;\n"
"}")
        self.ConfirmButton.setObjectName("ConfirmButton")
        self.CancelButton = QtWidgets.QPushButton(Dialog)
        self.CancelButton.setGeometry(QtCore.QRect(390, 240, 120, 50))
        self.CancelButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-radius:18px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:16pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:20px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-width: 3px;\n"
"border-radius:18px;\n"
"}")
        self.CancelButton.setObjectName("CancelButton")
        self.FormLabel = QtWidgets.QLabel(Dialog)
        self.FormLabel.setGeometry(QtCore.QRect(590, 170, 61, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.FormLabel.setFont(font)
        self.FormLabel.setStyleSheet("QLabel\n"
"{\n"
"font:8pt \"微软雅黑\";\n"
"color:#FF0000;\n"
"border-radius:3px;\n"
"}")
        self.FormLabel.setText("")
        self.FormLabel.setObjectName("FormLabel")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        #self.SaveRadioButton.clicked.connect(self.Function_SaveRadioButton)
        self.SaveToolButton.clicked.connect(self.Function_SaveToolButton)
        self.ConfirmButton.clicked.connect(self.Function_ConfirmButton)
        self.CancelButton.clicked.connect(self.Function_CancelButton)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "设置"))
        self.SaveRadioButton.setText(_translate("Dialog", "保存识别结果"))
        self.label.setText(_translate("Dialog", "选取输出数据保存文件夹"))
        self.SaveToolButton.setText(_translate("Dialog", "..."))
        self.label_3.setText(_translate("Dialog", "摄像头编号"))
        self.ConfirmButton.setText(_translate("Dialog", "确定"))
        self.CancelButton.setText(_translate("Dialog", "取消"))


    #数据输出文件夹按钮
    def Function_SaveToolButton(self):
        self.save_info = QFileDialog.getExistingDirectory(
            QtWidgets.QMainWindow(), '选择文件夹', '',) #save_info 统一数据输出保存文件夹
        self.SaveFolder_ShowStr(self.save_info)
        #执行函数
    
    #确认按钮
    #点击确认后设置的所有变量将会保存为全局变量并且保存在settings.txt文件中以便下一次调用
    #保存了三个数据 分别是 全局数据输出文件夹路径 是否进行数据保存状态 摄像头编号 按行保存
    #gol_savepath（字符串） gol_savestate（布尔） gol_cameraid（字符串）
    def Function_ConfirmButton(self):
        if self.CameraNumCheck():
            self.SaveLineEdit.setText(QtCore.QCoreApplication.translate("Dialog",self.SaveLineEdit.text()))
            settings = open("settings.txt", "r+")
            settings.truncate()
            savedata = [self.SaveLineEdit.text()+'\n',str(self.SaveRadioButton.isChecked())+'\n',self.CameraNumLineEdit.text()+'\n',]
            settings.writelines(savedata)
            settings.close()
            global gol_savepath
            global gol_savestate
            global gol_cameraid
            gol_savepath = self.SaveLineEdit.text()
            gol_savestate = self.SaveRadioButton.isChecked()
            gol_cameraid = self.CameraNumLineEdit.text()
            bl.gol_savepath = gol_savepath
            bl.gol_savestate = gol_savestate
            dialog_settings.close()
        else:
            self.StateTextCheck("格式错误")
        
    #取消按钮,取消识别,这个函数中的代码暂时只承担了关闭当前窗口的功能，
    #点击取消后具体要执行的任务代码请加在函数内
    def Function_CancelButton(self):
        #执行函数
        dialog_settings.close()

    #输出数据保存路径显示
    def SaveFolder_ShowStr(self,folderstr):
        self.SaveLineEdit.setText(QtCore.QCoreApplication.translate("Dialog", folderstr))
    
    #摄像头编号格式检查
    def CameraNumCheck(self):
        num_str = self.CameraNumLineEdit.text()
        timer = 0
        if len(num_str) == 0:
            return True

        for s in num_str:
            if (s == ',') or (s >= '0' and s <= '9'):
                if s == ',' and timer == 0:
                    return False
                elif s == ',':
                    timer = 0
                else:
                    timer = 1
            else:
                return False

        if num_str[-1] == ',':
            return False

        return True

    #摄像头编号字符串检测后状态输出，若字符串格式不对则会输出格式错误
    def StateTextCheck(self,state_str):
        self.FormLabel.setText(QtCore.QCoreApplication.translate("Dialog", state_str))
    
    #设置数据重装载
    def Settings_Load(self):
            settings = open("settings.txt", "r")
            loaddata = settings.readlines()
            self.SaveLineEdit.setText(QtCore.QCoreApplication.translate("Dialog",loaddata[0][0:-1]))
            if loaddata[1][0:-1] == 'True':
                self.SaveRadioButton.setChecked(True)
            else:
                self.SaveRadioButton.setChecked(False)
            self.CameraNumLineEdit.setText(QtCore.QCoreApplication.translate("Dialog",loaddata[2][0:-1]))
            settings.close()  

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setBaseSize(QtCore.QSize(0, 0))
        MainWindow.setStyleSheet("QMainWindow\n"
"{\n"
"background:url(:/background/background.png)\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.LocalVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.LocalVideoButton.setGeometry(QtCore.QRect(80, 330, 270, 90))
        self.LocalVideoButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:25px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#10c0ff;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}")
        self.LocalVideoButton.setObjectName("LocalVideoButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(20, 10, 401, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("QLabel\n"
"{\n"
"color:#D4007A;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"}")
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.BatchPictureButton = QtWidgets.QPushButton(self.centralwidget)
        self.BatchPictureButton.setGeometry(QtCore.QRect(30, 210, 270, 90))
        self.BatchPictureButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:25px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}")
        self.BatchPictureButton.setObjectName("BatchPictureButton")
        self.SinglePictureButton = QtWidgets.QPushButton(self.centralwidget)
        self.SinglePictureButton.setGeometry(QtCore.QRect(80, 80, 270, 90))
        self.SinglePictureButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:25px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#9EC4F3;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}")
        self.SinglePictureButton.setObjectName("SinglePictureButton")
        self.CameraVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.CameraVideoButton.setEnabled(True)
        self.CameraVideoButton.setGeometry(QtCore.QRect(30, 470, 270, 90))
        self.CameraVideoButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:25px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#10c0ff;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}")
        self.CameraVideoButton.setObjectName("CameraVideoButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(750, 100, 41, 361))
        self.label_2.setStyleSheet("QLabel\n"
"{\n"
"color:#0F224E;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"}")
        self.label_2.setObjectName("label_2")
        self.SettingsButton = QtWidgets.QPushButton(self.centralwidget)
        self.SettingsButton.setGeometry(QtCore.QRect(680, 515, 111, 61))
        self.SettingsButton.setStyleSheet("QPushButton\n"
"{\n"
"background:#00b0f0;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:25px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"background:#10c0ff;\n"
"color:#000000;\n"
"font:18pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"background:#8EB4E3;\n"
"color:#000000;\n"
"font:20pt \"微软雅黑\";\n"
"font-weight:bold;\n"
"border-style:ridge;\n"
"border-width: 3px;\n"
"border-color:#ffffff;\n"
"border-radius:30px;\n"
"}")
        self.SettingsButton.setObjectName("SettingsButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.SinglePictureButton.clicked.connect(self.Function_SinglePictureButton)
        self.BatchPictureButton.clicked.connect(self.Function_BatchPictureButton)
        self.LocalVideoButton.clicked.connect(self.Function_LocalVideoButton)
        self.CameraVideoButton.clicked.connect(self.Function_CameraVideoButton)
        self.SettingsButton.clicked.connect(self.Function_SettingsButton)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "安全帽识别系统"))
        self.LocalVideoButton.setText(_translate("MainWindow", "本地视频识别"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:26pt;\">安 全 帽 识 别 系 统</span></p></body></html>"))
        self.BatchPictureButton.setText(_translate("MainWindow", "批量图片识别"))
        self.SinglePictureButton.setText(_translate("MainWindow", "单独图片识别"))
        self.CameraVideoButton.setText(_translate("MainWindow", "实时视频识别"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>以</p><p>帽</p><p>取</p><p>人</p><p>团</p><p>队</p></body></html>"))
        self.SettingsButton.setText(_translate("MainWindow", "设置"))


    #单独图片是被按钮，打开文件选择窗口 
    def Function_SinglePictureButton(self):
        picsingle_file_info = QFileDialog.getOpenFileName(
        QtWidgets.QMainWindow(), '选择图片文件', '', '图片(*.jpg , *.png)')
        if picsingle_file_info[0] == '':  #如果没有选择图片而是取消 则return
            return
        picsingle_path = picsingle_file_info[0]  #picsingle_path,图片路径
        picture_op(picsingle_path, args, sess, boxes, scores, labels, input_data,)
        #执行函数

    #批量图片识别按钮，承担下一级窗口打开的功能    
    def Function_BatchPictureButton(self):
        global dialog_picfolder 
        dialog_picfolder = QDialog()
        child_picfolder_ui = Ui_PicFolderSelect_Dialog()  #批量图片处理窗口的实例化，要启动功能调用方法内函数
        child_picfolder_ui.setupUi(dialog_picfolder)
        dialog_picfolder.setWindowModality(Qt.ApplicationModal)
        dialog_picfolder.show()
        dialog_picfolder.exec()
        #执行函数
        
    #本地视频识别按钮，打开文件选择窗口
    def Function_LocalVideoButton(self):
        locvedio_file_info = QFileDialog.getOpenFileName(
        QtWidgets.QMainWindow(), '选择视频文件', '', '视频(*.mp4 , *.avi , *.mov , *.flv)')
        if locvedio_file_info[0] == '':  #如果没有选择视频而是取消 则return
            return
        locvedio_path = locvedio_file_info[0]  #locvedio_path,视频路径
        vedio_op(locvedio_path, args, sess, boxes, scores, labels, input_data)
        #执行函数
    
    #实时视频识别按钮，自行添加相关功能代码
    def Function_CameraVideoButton(self):
        cam_list = gol_cameraid.split(',')
        cam_list = [int(x) for x in cam_list]
        cam_op_new(args,sess,boxes, scores, labels,input_data,cam_list)
        #执行函数

    #设置按钮，打开下一级窗口
    def Function_SettingsButton(self):
        global dialog_settings
        dialog_settings = QDialog()
        child_settings_ui = Ui_Settings_Dialog()  #批量图片处理窗口的实例化，要启动功能调用方法内函数
        child_settings_ui.setupUi(dialog_settings)
        child_settings_ui.Settings_Load()
        dialog_settings.setWindowModality(Qt.ApplicationModal)
        dialog_settings.show()
        dialog_settings.exec()

    #主窗口实例化时调用此方法，用于读取设置文件中的数据赋值给全局变量 
    #gol_savepath（字符串） gol_savestate（布尔） gol_cameraid（字符串）
    def Global_DataSave(self):
        global gol_savepath
        global gol_savestate
        global gol_cameraid
        settings = open("settings.txt", "r")
        loaddata = settings.readlines()
        gol_savepath = loaddata[0][0:-1]
        if loaddata[1][0:-1] == 'True':
            gol_savestate = True
        else:
            gol_savestate = False
        gol_cameraid = loaddata[2][0:-1]
        bl.gol_savepath = gol_savepath
        bl.gol_savestate = gol_savestate
        settings.close()

if __name__ == "__main__":
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.6,
                                        nms_thresh=0.5)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        ui.Global_DataSave()
        MainWindow.show()
        sys.exit(app.exec_())

