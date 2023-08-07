# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:47:11 2021

@author: juano
"""
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from UVVIS_ThreadRPI import *
from UVVIS_GUI import *
import pandas as pd
import config
from gtts import gTTS
import pygame
from io import BytesIO
import math


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)
        self.bn_home.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.bn_VIPerson.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.bn_logItemRegister.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        #self.bn_bug.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        #Callbacks button into VIPerson configuration GUI
        self.pushButton_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.Next_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.Next_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        self.Back_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.Back_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        
        self.Preview.clicked.connect(lambda: self.ActivateCamera())
        self.pushButton_3.clicked.connect(lambda: self.Next_Config())
        self.Back_1.clicked.connect(lambda: self.ActivateOtherTab())
        self.pushButton_2.clicked.connect(lambda: self.open_dialog_box())
        self.Rectification_2.clicked.connect(lambda: self.RectificactionCamera())
        self.Disparity_Map_Bt.clicked.connect(lambda: self.showmapadisparidad())
        self.Depth_Angle_play.clicked.connect(lambda: self.Distance_SoundPrueba())
        self.Depth_Angle_play_2.clicked.connect(lambda: self.Inference_PreviewPrueba())
        self.Depth_Angle_play_3.clicked.connect(lambda: self.Local_MapLog())
        self.Accept_2.clicked.connect(lambda: self.ActivateUDV())
        self.ShowImageOnInterface = ShowImageOnInterface(" ", False)
        self.ShowPreviewMap = ShowPreviewMap(" ", False)
        self.ShowDepthMap = ShowDepthMap()
        self.Preview_camera.mousePressEvent = self.CalculateDepth
        self.path="SN24105.conf"
        self.Inference=False
        self.Sound=False
        self.DVU=False
        self.list_things=[]
        self.list_class=[]
        self.list_distancia=[]
        self.list_orientacion=[]
        self.list_comandos=[]
        
    def CalculateDepth(self, event):
        config.x = event.pos().x()
        config.y = event.pos().y()   
        
    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        self.path = filename[0]
        self.Open_rectificationfile.setText(self.path)
    
    def ImageUpdateLeftRect(self, Image):
        self.left_rect=Image
        
    def ImageUpdateRightRect(self, Image):
        self.right_rect=Image
        
    def ActivateCamera(self):
        self.ActivateRectification=False
        self.ShowImageOnInterface = ShowImageOnInterface("", self.ActivateRectification)
        if self.ShowImageOnInterface.isFinished:
            self.ShowImageOnInterface.start()
            self.ShowImageOnInterface.ImageUpdate.connect(self.ImageUpdateSlot)
            self.ShowImageOnInterface.ImageUpdate1.connect(self.ImageUpdateSlot1)
        
    def RectificactionCamera(self):
        self.ActivateRectification=True
        self.ShowImageOnInterface = ShowImageOnInterface(self.path,self.ActivateRectification)
        self.Disparity_Map_Bt.setEnabled(True)
        if self.ShowImageOnInterface.isFinished:
            self.ShowImageOnInterface.start()       
            self.ShowImageOnInterface.ImageUpdate.connect(self.ImageUpdateSlot)
            self.ShowImageOnInterface.ImageUpdate1.connect(self.ImageUpdateSlot1)
            self.ShowImageOnInterface.ParamsLabels.connect(self.UpdateParam)
    
    def showmapadisparidad(self):
        if self.ShowDepthMap.isFinished:
            self.ShowDepthMap.start()
            self.ShowDepthMap.ImageUpdate.connect(self.ImageUpdateSlotDepth)
            self.pushButton_3.setEnabled(True)
                
    def ImageUpdateSlot(self, Image):
        self.left_camera.setPixmap(QPixmap.fromImage(Image))
        
    def ImageUpdateSlot1(self, Image):
        self.right_camera.setPixmap(QPixmap.fromImage(Image))

    def UpdateParam(self, Parameter):
        self.Cx_left_camera.setText(str(Parameter[0][0]))
        self.Cy_left_camera.setText(str(Parameter[0][1]))
        self.Fx_left_camera.setText(str(Parameter[0][2]))
        self.Fy_left_camera.setText(str(Parameter[0][3]))
        self.K1_left_camera.setText(str(Parameter[0][4]))
        self.K2_left_camera.setText(str(Parameter[0][5]))
        self.P1_left_camera.setText(str(Parameter[0][6]))
        self.P2_left_camera.setText(str(Parameter[0][7]))
        self.K3_left_camera.setText(str(Parameter[0][8]))

        self.Cx_right_camera.setText(str(Parameter[1][0]))
        self.Cy_right_camera.setText(str(Parameter[1][1]))
        self.Fx_right_camera.setText(str(Parameter[1][2]))
        self.Fy_right_camera.setText(str(Parameter[1][3]))
        self.K1_right_camera.setText(str(Parameter[1][4]))
        self.K2_right_camera.setText(str(Parameter[1][5]))
        self.P1_right_camera.setText(str(Parameter[1][6]))
        self.P2_right_camera.setText(str(Parameter[1][7]))
        self.K3_right_camera.setText(str(Parameter[1][8]))

        self.Baseline_depth.setText(str(Parameter[2][0][0]))
        self.Ty_depth.setText(str(Parameter[2][0][1]))
        self.Tz_depth.setText(str(Parameter[2][0][2]))
        self.Rx_depth.setText(str(Parameter[2][1][0]))
        self.Cv_depth.setText(str(Parameter[2][1][1]))
        self.Rz_depth.setText(str(Parameter[2][1][2]))
        
    def ImageUpdateSlotDepth(self, Image):
        self.disparity_map.setPixmap(QPixmap.fromImage(Image))
    
    def ImageUpdatePreview(self, Image):
        self.Preview_camera.setPixmap(QPixmap.fromImage(Image))
    
    def Next_Config(self):
        config.ViewActivate=1
        self.ShowImageOnInterface.stop()
        self.ShowDepthMap.stop()
        """self.ShowPreviewMap = ShowPreviewMap(self.path,self.ActivateRectification)
        if self.ShowPreviewMap.isFinished:
            self.ShowPreviewMap.start()
            config.ViewActivate=1
            self.ShowPreviewMap.ImageUpdate.connect(self.ImageUpdatePreview)"""

    def ActivateOtherTab(self):
        config.ViewActivate=0
    
    def disparityList(self, Disparity_list):
        if config.ViewActivate==2:
            if (Disparity_list[5] > 0):
                depth = (Disparity_list[0]/1.6) * (-Disparity_list[2] / Disparity_list[5])
                changeInX = (Disparity_list[3]/1.6) - Disparity_list[6][0]
                changeInY = (Disparity_list[4]/1.6) - Disparity_list[6][1]
                theta_angle= np.degrees(math.atan2(changeInY,changeInX))
            else:
                depth = 0
                theta_angle=0
            self.depth_meter.setText('{0:.2f}'.format(depth / 1000) + "m")
            self.Angle_Alpha.setText('{0:1d}'.format(int(theta_angle)))
            gtts=gTTS (text = "Profundidad " + '{0:.2f}'.format(depth / 1000) + "m"+" Angulo delta: "+'{0:1d}'.format(int(theta_angle)), lang='es', slow=False)
            self.textTovoice(gtts)
    
    def pandas_to_str(self,clase,distancia,orientacion,position_commands):
        df = pd.DataFrame({ 
            'Clase' : clase,
            'Distancia' : distancia,
            'Orientación' : orientacion,
            'Posición' : position_commands})
        return df.to_string(col_space =14,justify = "justify")

    def Distance_SoundPrueba(self):
        #self.ShowPreviewMap.start()
        config.ViewActivate=2
        self.Sound=True
        if(self.Sound & self.Inference):
            self.Next_1.setEnabled(True)
        #self.ShowDepthMap.disparityLog.connect(self.disparityList)
        
    def Inference_PreviewPrueba(self):
        self.Inference=True
        self.Sound=True
        self.ShowPreviewMap.stop()
        if(self.Sound & self.Inference):
            self.Next_1.setEnabled(True)
        self.ShowInferenceModel = ShowInferenceModel(self.path,self.ActivateRectification)
        if self.ShowInferenceModel.isFinished:
            config.ViewActivate=3
            self.ShowInferenceModel.start()
            self.ShowInferenceModel.ObjectsDetect.connect(self.UpdateinferenceList)
            self.ShowInferenceModel.ImageUpdate.connect(self.Updateinference)

    def Updateinference(self, Image):
        if config.ViewActivate==3:
            self.Preview_camera_2.setPixmap(QPixmap.fromImage(Image))
        elif config.ViewActivate==4:
            self.Preview_camera_3.setPixmap(QPixmap.fromImage(Image))

    def UpdateinferenceList(self, ListDetected):
        if config.ViewActivate==3:
            self.class_label.setText(str(ListDetected[0][:]))
            self.score.setText(str(ListDetected[1]))
        elif config.ViewActivate==4:
            position_commands=''
            if (ListDetected[3] > 0):
                depth = (float(self.Fx_left_camera.text())/1.6) * -(float(self.Baseline_depth.text()) / ListDetected[3])
                changeInX = (float(self.Cx_left_camera.text())/1.6)   - ListDetected[2][0]
                changeInY = (float(self.Cy_left_camera.text())/1.6)   - ListDetected[2][1]
                theta_angle= np.degrees(math.atan2(changeInY,-changeInX))
            else:
                depth = 0
                theta_angle=0
                position_commands='no registra'
            if theta_angle>=-15 and theta_angle<15:
                position_commands='hacia la derecha'
            elif theta_angle>=15 and theta_angle<75:
                position_commands='hacia la diagonal derecha'
            elif theta_angle>=165 and theta_angle<-165:
                position_commands='la izquierda'
            elif theta_angle>=105 and theta_angle<165:
                position_commands='la diagonal izquierda'
            elif (theta_angle>=75 and theta_angle<105):
                position_commands='adelante'
            elif theta_angle>=-75 and theta_angle<-15:
                position_commands='la diagonal derecha abajo'
            elif theta_angle>=-165 and theta_angle<-105:
                position_commands='la diagonal izquierda abajo'
            elif theta_angle>=-105 and theta_angle<75:
                position_commands='adelante abajo'
            self.ObjectReconice_log_2.setStyleSheet('color: white')
            self.ObjectReconice_log_2.setFont(QtGui.QFont("Monospace"))
            self.list_class.append(ListDetected[0])
            self.list_distancia.append('{0:.2f}'.format(depth/1000))
            self.list_orientacion.append(theta_angle)
            self.list_comandos.append(position_commands)
            if (len(self.list_class)==10):
                self.list_things = self.pandas_to_str(self.list_class, self.list_distancia, self.list_orientacion, self.list_comandos)
                if self.DVU:
                    self.list_things.sort_values(by=['Distancia'], inplace=True)
                    for i in range(len(self.list_things)):
                        config.gtts=gTTS (text = "El objeto "+self.list_things['Clase'].values[i]+"esta a " + '{0:.2f}'.self.list_things['Distancia'].values[i] + "m, "+self.list_things['Posición'].values[i], lang='es', slow=False)
                self.list_class=[]
                self.list_distancia=[]
                self.list_orientacion=[]
                self.list_comandos=[]
            self.ObjectReconice_log_2.setText(self.list_things)
            self.ObjectReconice_log_2.showMaximized()
            
        
    def Local_MapLog(self):
        config.ViewActivate=4
        if(self.Sound & self.Inference):
            self.Accept_2.setEnabled(True)
        self.ShowInferenceModel.ObjectsDetect.connect(self.UpdateinferenceList)
        self.ShowInferenceModel.ImageUpdate.connect(self.Updateinference)
    
    def ActivateUDV(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowTitle("Confirmación de cambio de modo")
        msgBox.setText("¿Esta seguro de cambiar al modo de usuario con discapacidad visual?")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.DVU=True
            self.lab_user.setText("Usuario con Discapacidad Visual")
            self.lab_person.setPixmap(QtGui.QPixmap("Images/65734.png"))
            self.lab_person.setScaledContents(True)
            
    def textTovoice(self,tts) :
        # convert to file-like object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
            #--- play it ---
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.set_volume(self.dialAudioLevel.value()/100)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()