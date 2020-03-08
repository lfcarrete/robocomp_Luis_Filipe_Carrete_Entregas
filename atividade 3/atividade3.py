import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
import statistics

def reta (x1,y1,x2,y2):
    m = (x2 - x1)/(y2 - y1)
    
    y = m * x2 - m * x1 + y2
    
    return y
    

def fuga (img):
    
   # img = cv2.imread(img)

    teste_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mascara = cv2.inRange(teste_gray,250,255)
    
    # Apply edge detection method on the image 
    edges = cv2.Canny(mascara,100,255,apertureSize = 5) 

    # Detect points that form a line

    lines = cv2.HoughLinesP(edges, 10, math.pi/180.0, 100, np.array([]), 45, 5)

    # Draw lines on the image

    for line in lines:

        x1, y1, x2, y2 = line[0]
        
        
        cv2.line(img, (x1,y1), (x2, y2), (255, 0, 0), 3)
        
    #Show result
    return(img)
    

#plt.imshow(fuga('teste.png'))
    
video_capture = cv2.VideoCapture("vid1.mp4")
while True:
    ret, frame = video_capture.read()
    if ret:
        
        cv2.imshow('Video', fuga(frame))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
video_capture.release()
cv2.destroyAllWindows()









