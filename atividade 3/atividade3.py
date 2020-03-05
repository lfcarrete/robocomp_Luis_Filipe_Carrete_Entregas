import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
import statistics



def fuga (img):
    
    teste = cv2.imread(img)

    teste_gray = cv2.cvtColor(teste, cv2.COLOR_BGR2GRAY)

    mascara = cv2.inRange(teste_gray,250,255)

    #lines = cv2.HoughLinesP(mascara, 10, math.pi/180.0, 100, np.array([]), 45, 5)

    #a,b,c = lines.shape

    #for i in range(a):
        # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
     #   cv2.line(mascara, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 5, cv2.LINE_AA)
    
    plt.imshow(mascara)
    
    


fuga('teste.png')