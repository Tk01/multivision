import project
import visualize
import cv2
import cv2.cv as cv
import numpy as np
import math
def readRadiograph(graphNumber):
        if graphNumber < 10:
            img = cv2.imread("Radiographs/Cut_0"+str(graphNumber)+".tif",0)
        else:
            img = cv2.imread("Radiographs/Cut_"+str(graphNumber)+".tif",0)
            
        return img
def displayVectorizedEdgeData(image, vectorizedEdgeData,i):
    m,n = image.shape
    img = np.zeros((m,n,3), np.uint8)
    
    for (x,y) in vectorizedEdgeData:
        img[x,y] = (255,255,255)
    
    img2=cv2.resize(img,(300,150))
    return img2
    
def findVectorizedEdgeData(img,(x1,y1),(x2,y2)):
    #bovenkant
    filter_length = 5
    sigma = 1
   # result = cv2.bilateralFilter(img,12,17,17)
    result1 = cv2.adaptiveBilateralFilter(img,(13,13),13)  
    #cv2.imshow('img_res',result1)
    #cv2.waitKey(0)
    edges1 = cv2.Canny(np.uint8(result1), 1, 15,L2gradient=True)
    #onderkant
    filter_length = 5
    sigma = 1
    result = cv2.bilateralFilter(img  ,9,20,20)
    #cv2.imshow('img_res',result1)
    #cv2.waitKey(0)
    edges = cv2.Canny(np.uint8(result), 1, 20,L2gradient=True)
    #result = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)  
    mid = (y1 + y2 ) / 2
    
    edges[0:mid][:] = edges1[0:mid][:]
    

    array = []
    [M,N] = np.shape(edges)
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            if y>=0 and y< M and x>=0 and x< N and edges[y,x] != 0:
                array.append((y,x))
    return array
    
if __name__ == '__main__':
    result = np.zeros((790,920,3),np.uint8)
    for i in range(1,15):
        img = readRadiograph(i)
        [M,N] = np.shape(img)
        array = findVectorizedEdgeData(img,(0,0),(N,M))
        img2=displayVectorizedEdgeData(img,array,i)
        result[math.floor(i/3)*160:math.floor(i/3)*160+150,(i%3)*310:(i%3)*310+300,:]=img2
    cv2.imshow('img_res',result)
    cv2.waitKey(0)