import cv2
import cv2.cv as cv
import numpy as np
import Image
import math
def getModelData():
    result= np.zeros((80,14))
    for i in range(14):
        f = open('Landmarks\original\landmarks'+str(i+1)+'-1.txt', 'r')
        t=0
        for j in f:
            result[t,i]=int(float(j.rstrip()))
            t = t+1
    return result
def reallign(data):
    for i in range(14):
        x=0
        y=0
        for j in range(40):
            x=x+data[j,i]
            y=y+data[j+40,i]
        for j in range(40):
            data[j,i]=data[j,i]-x/40
            data[j+40,i]=data[j+40,i]-y/40
    example=data[:,0]/np.linalg.norm(data[:,0])
    while True:
        examplestored=example
        for i in range(14):
            a=0
            b=0
            for j in range(40):
                a=a+data[j,i]*example[j]
                b=b+(example[j]*data[j+40,i]-example[j+40]*data[j,i])
            s=math.sqrt(a*a+b*b)
            t=math.atan(b/a)
            for j in range(40):
                data[j,i]= math.cos(t)*data[j,i]-math.sin(t)*data[j+40,i]
                data[j+40,i]= math.sin(t)*data[j,i]+math.cos(t)*data[j+40,i]
            data[:,i]=data[:,i]/s
        example =np.sum(data,axis=1)/14
        meanx=0
        meany=0
        for j in range(40):
                meanx=meanx+example[j]
                meany=meany+example[j+40]
        for j in range(40):
                example[j]=example[j]-meanx/40
                example[j+40]=example[j+40]-meany/40
        example=example/np.linalg.norm(example)
        if max(abs(example-examplestored))<0.001 :
            break 
    return data
def generateModel(data):
    return    
def getTestData():
    return
def fit(data,model):
    return
def PCA(pcaData):
    return
if __name__ == '__main__':
    data = getModelData()
    reallignedData = reallign(data)
    pcaData = PCA(reallignedData)
    model = generateModel(pcaData)
    testData = getTestData()
    result = fit(testData,model)
    
        
        