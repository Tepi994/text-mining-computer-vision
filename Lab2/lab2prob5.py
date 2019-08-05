# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:42:36 2019

@author: jctep
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def convolution(img, kernel):
    
#    if kernel.shape[0] != kernel.shape[1]:
#        return('Please use squared matrix for the kernel')

    if kernel.shape[0] % 2 != 0 and kernel.shape[1] % 2 != 0:
        print("okay")
        
    else:
        print("please introduce a square and odd matrix")
        
    #elif (kernel.shape[0]%2) == 0:
    #    return('Please use an odd dimension for the kernel')
    
    image = cv2.imread(img)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height = image_gray.shape[0]
    width = image_gray.shape[1]
    
    init = int(np.floor(kernel.shape[0]/2))
    
    new_image = np.array([])
    
    for i in range(init, width-init-1):
        for j in range(init, height-init-1):
            new_matrix = image_gray[i-int(np.floor((kernel.shape[0])/2)):i+int(np.floor((kernel.shape[0])/2)+1),
                                 j-int(np.floor((kernel.shape[0])/2)):j+int(np.floor((kernel.shape[0])/2)+1)]
            value = (np.sum(new_matrix*kernel))/(kernel.shape[0]*kernel.shape[1])
            new_image = np.append(new_image, value)
    new_image = new_image.reshape((height-kernel.shape[0], width-kernel.shape[0]))
    cv2.imwrite('conv_out.jpg', new_image)
    return(new_image)
    
k = np.array([[1,2,1],[2,4,2],[1,2,1]])

convolution("face2.jpg",k)



