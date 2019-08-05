# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:39:59 2019

@author: jctep
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def transformation(img, alpha = 1, beta = 1, gamma = 1, transformation ="lineal"):
    image_0 = cv2.imread(img)
    image = cv2.cvtColor(image_0,cv2.COLOR_BGR2GRAY)
    height = image.shape[0]
    width = image.shape[1]
    
    
    if transformation == "negative":
        
    
    ### negative image


        negative_image = np.zeros((height,width)) + 256

        negative_image =  (256 - 1) - image
        print("negative")
        cv2.imwrite("image_neg.jpg",negative_image)
        image_neg = cv2.imread("image_neg.jpg")     
        cv2.imshow('Negative Image',image_neg)
    
   
    elif transformation == "multiplication":
        
    #multiplication 
    
        mult_image = alpha*image

        cv2.imwrite("image_mult.jpg",mult_image)
        print("multiplication")
        mult_image = cv2.imread("image_mult.jpg")     
        cv2.imshow('Multiplication of  Image',mult_image)
    
    elif transformation == "division":
        

    #### division
        div = 1/beta
        div_image = div*image 
        cv2.imwrite("image_div.jpg",div_image)
        print("division")
        div_image = cv2.imread("image_div.jpg")     
        cv2.imshow('Division of  Image',div_image)
    
    
    ### lineal transformation 
    
    elif transformation =="lineal":
        

        lin_image = alpha*image + beta
        cv2.imwrite("image_lin.jpg",lin_image)
        lin_image = cv2.imread("image_lin.jpg")  
        print("lineal")
        cv2.imshow('Lineal Transformation of Image',lin_image)
    
    elif transformation =="log":
        

    ## logistic transformation 

        log_image = alpha*np.log(np.ones((height,width)) + image)
        cv2.imwrite("image_log.jpg",log_image)
        print("log")
        log_image = cv2.imread("image_log.jpg")     
        cv2.imshow('Log Transformation of Image',log_image)
        
    elif transformation == "exponential":
        

    
     ## exponential transformation 
    
        exp_image = alpha*((image)**gamma)
        print("exponential")
        cv2.imwrite("image_exp.jpg",exp_image)
        exp_image = cv2.imread("image_exp.jpg")     
        cv2.imshow('Exponential Transformation of Image',exp_image)
    
    else:
        print("please select a valid transformation parameter.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
    
    
transformation("beatles.jpg", alpha = 1.5 , beta = 1.2, transformation="lineal")
transformation("beatles.jpg",transformation="negative")
transformation("beatles.jpg", alpha = 1.3,transformation="multiplication")
transformation("beatles.jpg", beta = 1.3,transformation="division")
transformation("beatles.jpg", alpha = 10 ,transformation="log")
transformation("beatles.jpg", alpha = 10 ,transformation="multinomial")

  