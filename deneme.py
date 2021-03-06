# -*- coding: cp1254 -*-
from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import glob
import os
import gc
from sklearn.externals import joblib
from threading import Thread
from Queue import Queue
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import argparse
import imutils
import os


resimler = "C:\\Users\\Z�YA\\Desktop\\Datasets\\Dataset-3"

datas = []

def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cvtColor(image, COLOR_BGR2HSV)
    hist = calcHist([hsv], [0, 1, 2], None, bins,
            [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
            hist = normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
            normalize(hist, hist)

    # return the flattened histogram as the feature vector

    return hist.flatten()


def colorSeg(hsv_image):
    lowerAndUpperArray = np.matrix([[170,128,128],[180,255,255],[1,128,128],[30,255,255],[40,128,100],[95,255,255],[20,128,128],[30,255,255]])

    red1_mask = inRange(hsv_image,lowerAndUpperArray[0,:],lowerAndUpperArray[1,:])
    red2_mask = inRange(hsv_image,lowerAndUpperArray[2,:],lowerAndUpperArray[3,:])
    green_mask = inRange(hsv_image,lowerAndUpperArray[4,:],lowerAndUpperArray[5,:])
    yellow_mask = inRange(hsv_image,lowerAndUpperArray[6,:],lowerAndUpperArray[7,:])

    binaryImage = red1_mask+red2_mask+green_mask+yellow_mask

    return binaryImage

def imageBlur(crop_image):
    median = medianBlur(crop_image,9)
    return median

def imageMorp(binaryImage):
    kernel = getStructuringElement(MORPH_ELLIPSE,(5,5))
    dilation = dilate(binaryImage,kernel,iterations=1)
    return dilation

def findCircleDraw(canny_image,image,model):
    redCounter = 0
    yellowCounter = 0
    greenCounter = 0
    
    circles = HoughCircles(canny_image,HOUGH_GRADIENT,2,40,param1=50,param2=20,minRadius=5,maxRadius=30)
    drawCircleImage = image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            
            #rectangle(drawCircleImage,(i[0]-i[2],i[1]-i[2]),(i[0]+i[2],i[1]+i[2]),(255,0,0),1) 
            im = drawCircleImage[(i[1]-i[2]):(i[1]+i[2]),(i[0]-i[2]):(i[0]+i[2])]
            imre_hist = extract_color_histogram(im)
            datas.append(imre_hist)
            deger = model.predict(datas)
            if deger == 0:
                greenCounter = greenCounter+1
                #circle(image,(450,481),20,(0,255,0),-1)
            elif deger == 1:
                redCounter = redCounter+1
                #circle(image,(450,481),20,(0,0,255),-1)
            elif deger == 2:
                yellowCounter = yellowCounter+1
                #circle(image,(450,481),20,(0,255,255),-1)
            else:
                print "Yok :D"
            
            datas.pop()
            #print model.predict(datas)
        if greenCounter!=0:
            circle(image,(450,481),20,(0,255,0),-1)
        elif redCounter>yellowCounter:
            circle(image,(450,481),20,(0,0,255),-1)
        else:
            circle(image,(450,481),20,(0,255,255),-1)
            
    return drawCircleImage
    

if __name__ == "__main__":
    gc.enable()
    basla = time.time()
    cap = VideoCapture('output.avi')

    while(True):
        ret, image = cap.read()
        crop_image = image[0:(image.shape[0]/2),0:(image.shape[1])]
        median = imageBlur(crop_image)
        hsv_image = cvtColor(median,COLOR_BGR2HSV)

        binaryImage = colorSeg(hsv_image)
        dilation = imageMorp(binaryImage)

        canny = Canny(dilation,75,150)
        model = joblib.load("D:\\svm.model")
        drawCircleImage = findCircleDraw(canny,image,model)
        
        imshow('frame',image)
        if waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print "%s saniye s�rd�" % (time.time() - basla)
    
