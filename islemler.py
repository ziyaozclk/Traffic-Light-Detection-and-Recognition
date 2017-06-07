# -*- coding: cp1254 -*-
from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import glob
import os
import gc
from threading import Thread
from Queue import Queue



def boyutDegistir(resimlerPath):
    sayac = 0
    for resim in glob.glob(os.path.join(resimlerPath,"*.png")):
        image = imread(resim)
        image = resize(image,(900,506))
        imwrite(resimlerPath+str(sayac)+".png",image)
        sayac = sayac+1

def veriTasi(srcPath,toPath):
    sayac = 0
    for resim in glob.glob(os.path.join(srcPath,"*.png")):
        image = imread(resim)
        imwrite(toPath+str(sayac)+".png",image)
        sayac = sayac+1

def videoOkuma(videoPath):
    cap = VideoCapture(videoPath)
    sayac = 0
    while(True):
        ret, frame = cap.read()
        frame = resize(frame,(900,506))
        imwrite("C:\\Users\\ZİYA\\Desktop\\Datasets\\Dataset-3\\"+str(sayac)+".png",frame)
        sayac = sayac+1
    cap.release()
    destroyAllWindows()

    
if __name__ == "__main__":
    resimler = "C:\\Users\\ZİYA\\Desktop\\Datasets\\Dataset-2\\"
    RED = "C:\\Users\\ZİYA\\Desktop\\Datasets\\red\\"
    GREEN = "C:\\Users\\ZİYA\\Desktop\\Datasets\\green\\"
    YELLOW = "C:\\Users\\ZİYA\\Desktop\\Datasets\\yellow\\"
    TRAIN = "C:\\Users\\ZİYA\\Desktop\\Datasets\\train\\"
    
    #boyutDegistir(resimler)
    #veriTasi(RED,TRAIN)
    videoOkuma("C:\\Users\\ZİYA\\Desktop\\Datasets\\3.mp4")
