# -*- coding: cp1254 -*-
import numpy as np
from cv2 import *
import math
import time
import glob
import os

resimler = "C:\\Users\\ZİYA\\Desktop\\Datasets\\Dataset-2"


fourcc = VideoWriter_fourcc(*'XVID')
out = VideoWriter('3.avi',fourcc, 30.0, (900,506))

sayac = 0
for resim in glob.glob(os.path.join(resimler,"*.png")):
    frame = imread("C:\\Users\\ZİYA\\Desktop\\Datasets\\Dataset-2\\"+str(sayac)+".png")
    out.write(frame)
    imshow('frame',frame)
    sayac = sayac+1
    if waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
