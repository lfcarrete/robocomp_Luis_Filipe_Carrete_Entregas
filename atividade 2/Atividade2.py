# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:05:31 2020

@author: lfcsa
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Luis Filipe Carrete, Manuel Castanares, Gustavo Pierre"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to useq when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1





print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# Essa função vai ser usada abaixo. Ela encontra a matriz (homografia) 
# que quando multiplicada pela imagem de entrada gera a imagem de 

def find_homography_draw_box(kp1, kp2, img_cena):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()


    
    h,w = img_original.shape
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)


    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    
    circles = []


    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of magenta and blue color in HSV
    lower_mag = np.array([324/2,100,100])
    upper_mag = np.array([364/2,255,255])
    
    lower_blue = np.array([80,100,100])
    upper_blue = np.array([150,255,255])
    

    # Threshold the HSV image to get only magenta colors
    mask_mag = cv2.inRange(hsv, lower_mag, upper_mag)
    
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #Justa os masks que mostra o circulo azul e o circulo magenta
    res1 = cv2.bitwise_or(mask_blue,mask_mag)
    
    res_rgb = cv2.cvtColor(res1,cv2.COLOR_GRAY2RGB)
    
    logo_bgr = cv2.imread('logo_insper.png')
    

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    
    
    
    # Número mínimo de pontos correspondentes
    MIN_MATCH_COUNT = 10
    
    cena_bgr = frame # Imagem do cenario
    original_bgr = logo_bgr
    
    # Versões RGB das imagens, para plot
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    cena_rgb = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2RGB)
    
    # Versões grayscale para feature matching
    img_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    img_cena = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2GRAY)
    
    framed = None
    
    # Imagem de saída
    out = cena_rgb.copy()
    
    
    # Cria o detector BRISK
    brisk = cv2.BRISK_create()
    
    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)
    
    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    
    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    
    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        print("Matches found")    
        framed = find_homography_draw_box(kp1, kp2, cena_rgb)
        cv2.imshow("AAA",framed)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

    a = 0

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            #print(i)
        
            #Funcao que guarda o ponto do ultimo circulo
            
            if a == 0:
                b = i[0]
                c = i[1]
                a += 1
            else:
                a = 0
            #Desenha linha 
            cv2.line(res_rgb,(i[0],i[1]),(b,c),(255,0,0),5)
            
            #Calcula distancia do papel e da tela usando a calibracao
            dist = (3.1/i[2])*339.71
            #print("Distancia: {} cm".format(dist))
            # draw the outer circle
            cv2.circle(res_rgb,(i[0],i[1]),i[2],(0,0,255),3)
            
            # draw the center of the circle
            cv2.circle(res_rgb,(i[0],i[1]),2,(0,0,255),3)
            
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(res_rgb,(i[0],i[1]),i[2],(0,255,0),2)
            
        
            #Funcao que calcula Angulo
            delta_x = b - i[0]
            delta_y = c - i[1]
            #Gatante que a funcao nao pegue numeros negativos
            if delta_x < 0:
                delta_x *= -1
            if delta_y < 0:
                delta_y *= -1
            
            #Calcula hipotenusa
            hyp = (delta_x**2 + delta_y**2)**0.5
            #Calcula angulo e converte de rad para graus
            angulo = math.acos(delta_x/hyp) * (180/math.pi)
            
            #Escreve distancia na tela
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(res_rgb,"Distancia: {:.2f} cm".format(dist),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            
            #Escreve Angulo se valor for diferente de 0
            if angulo > 0:
                cv2.putText(res_rgb,"Angulo: {:.2f} deg".format(angulo),(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    
            
    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(bordas_color,(0,0),(511,511),(255,255,255),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    
    
    # Display the resulting frame
    cv2.imshow('Detector de circulos', res_rgb)

 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
