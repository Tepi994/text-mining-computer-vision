# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:53:45 2019

@author: jctep
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

imagen = cv2.imread("car.jpg")
imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

##preng biba
def convolutional_filter(imagen, kernel):
    alto = imagen.shape[0]
    ancho =  imagen.shape[1]
    
    altoKernel = len(kernel[:,0])
    anchoKernel = len(kernel[0,:])

    #parte entera del centro del kernel
    step = int(len(kernel[0,:])/2)

    newImg = np.zeros((alto, ancho, 1))
    
    for i in range(step, alto - step):
        newPixel = 0
        for j in range(step, ancho - step):
            ventana = imagen[i-step:i+2*step, j-step:j+2*step]
            newPixel = 0
            for m in range(0, altoKernel):
                for n in range(0, anchoKernel):
                    newPixel = newPixel + ventana[m, n] * kernel[m, n]
            newImg[i, j] = int(newPixel)
    
    return newImg

#usando la convolución.
kernelGauss_33 = np.array([[1/16., 2/16., 1/16.], [2/16., 4/16., 2/16.], [1/16., 2/16., 1/16.]])
filtrada = convolutional_filter(imagen, kernelGauss_33)
    
    
def derivadaX(imagen):
    #image = cv2.imread(img)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height = imagen.shape[0]
    width = imagen.shape[1]
    out = np.zeros((height,width,1))
    
    for i in range(0,height - 1):
        for j in range(0,width):
            out[i,j] = imagen[i+1,j] - imagen[i,j]
    
    return(out)
            
def derivadaY(imagen):
    #image = cv2.imread(img)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height = imagen.shape[0]
    width = imagen.shape[1]
    out = np.zeros((height,width,1))
    
    for i in range(0,height):
        for j in range(0,width-1):
            out[i,j] = imagen[i,j+1] - imagen[i,j]
    
    return(out)
            
def gradient(x, y):
    height = x.shape[0]
    width = x.shape[1]
    
    out = np.zeros((height, width, 1))

    for i in range(0, height):
        for j in range(0, width):
            out[i,j] = np.sqrt(x[i,j]**2 + y[i,j]**2)
    
    return(out)

def phase(derX,derY):
    height = derX.shape[0]
    width = derY.shape[1]
    out = np.zeros((height,width,1))
    

    
    for i in range(0, height):
        for j in range(0, width):
            out[i,j] = np.where((math.atan2(derY[i,j], derX[i,j]) * (180/np.pi))<0,
                               (math.atan2(derY[i,j], derX[i,j]) * (180/np.pi))+360,
                               (math.atan2(derY[i,j], derX[i,j]) * (180/np.pi)))
    
    return out

def non_max_supress(image,phase):
    #image = cv2.imread(img)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        

    
    
    height = image.shape[0]
    width = image.shape[1]
    out = np.zeros((height,width))
    
    for i in range(1, height - 2):
        for j in range(1, width -2):
            ang = phase[i,j]
            
            ### 337.5 - 22.5 or 157.5 -202.5
            if(((ang > 337.5 and ang <= 360) or (ang > 0 and ang <= 22.5)) or ( ang > 157.5 and ang <= 202.5)):
                
                if (image[i,j] > image[i,j+1]) and (image[i,j] > image[i,j-1]):
                    out[i,j] = image[i,j]
                else:
                    out[i,j] = 0

                        
            ###22.5 - 67.5 or 202.5 - 247.5
            elif((ang > 22.5 and ang <= 67.5) or ( ang > 202.5 and ang <= 247.5)):
                
                if (image[i,j] > image[i+1,j-1]) and (image[i,j] > image[i-1,j+1]):
                    out[i,j] = image[i,j]
                else:
                    out[i,j] = 0

                       ###67.5 - 112.5 or 247.5 - 292.5
            elif((ang > 67.5 and ang <= 112.5) or ( ang > 247.5 and ang <= 292.5)):
                
                if (image[i,j] > image[i-1,j]) and (image[i,j] > image[i+1,j]):
                    out[i,j] = image[i,j]
                else:
                    out[i,j] = 0
            
            
            ### 112.5 - 157.5 or 292.5 - 337.5
            elif((ang > 112.5 and ang <= 157.5) or ( ang > 292.5 and ang <= 337.5)):
                
                if (image[i,j] > image[i-1,j-1]) and (image[i,j] > image[i+1,j+1]):
                    out[i,j] = image[i,j]
                else:
                    out[i,j] = 0

    

 
    cv2.imwrite("ms.jpg",out)
    #canny = cv2.imread("cannycarnonconv.jpg")     
    #cv2.imshow('Resultado',canny)  


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #sys.exit()                    
          
    return(out)       
            
dX = derivadaX(filtrada)
dY = derivadaY(filtrada)
gradiente = gradient(dX,dY)
fase = phase(dX,dY)
nm = non_max_supress(gradiente,fase)

def binarization(image, threshold):

    
    height = image.shape[0]
    width = image.shape[1]
    
    out = np.where(image<=threshold,0,255)
    
    #writing image
    cv2.imwrite("binary_image.jpg", out)
    binary_image = cv2.imread("binary_image.jpg")
    
    
    #result image
    cv2.imwrite("resultado_final.jpg", binary_image)
    return(binary_image)

binary = binarization(nm, 12)

def main():
    imagen = cv2.imread("resultado_final.jpg")
    cv2.imshow('Canny Algorithm Image',imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
    
main()
         
            
            
    
    
    
                
  
                
            
            
    
