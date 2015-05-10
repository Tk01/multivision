import cv2
import cv2.cv as cv
import numpy as np
import Image
import math

#
# Landmarks zijn (x1,y1,x2,y2,...) geordend, zie visualize (heb normaal deze fout verbeterd)
#
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
    #1. Translate each example so that its centre of gravity is at the origin
    for i in range(14):
        x=0
        y=0
        for j in range(0,80,2):
            x=x+data[j,i]
            y=y+data[j+1,i]
        for j in range(0,80,2):
            data[j,i]=data[j,i]-x/40
            data[j+1,i]=data[j+1,i]-y/40
    #2. Choose one example as an initial estimate of the mean shape and scale so
    example=data[:,0]/np.linalg.norm(data[:,0])
    #3. Record the first estimate as x0 to define the default orientation.
    x0=example
    while True:
        examplestored=example
        #4. Align all the shapes with the current estimate of the mean shape.
        '''
        for i in range(14):
            a=0
            b=0
            for j in range(0,80,2):
                a=a+data[j,i]*example[j]
                b=b+(example[j]*data[j+1,i]-example[j+1]*data[j,i])
            s=math.sqrt(a*a+b*b)
            t=math.atan(b/a)
            for j in range(0,80,2):
                data[j,i]= math.cos(t)*data[j,i]-math.sin(t)*data[j+1,i]
                data[j+1,i]= math.sin(t)*data[j,i]+math.cos(t)*data[j+1,i]
            data[:,i]=data[:,i]/s
        '''
        for i in range(14):
            #4.1 |xj|=1
            data[:,i] = data[:,i]/np.linalg.norm(data[:,i])
            #4.2 find angle
            a=0
            b=0
            for j in range(0,80,2):
                a=data[j,i]*example[j+1]-data[j+1,i]*example[j]
                b=data[j,i]*example[j]+data[j+1,i]*example[j+1]
            angle=math.atan(a/b)
            #4.3 rotate
            for j in range(0,80,2):
                data[j,i]=math.cos(angle)*data[j,i]-math.sin(angle)*data[j+1,i]
                data[j+1,i]=math.sin(angle)*data[j,i]+math.cos(angle)*data[j+1,i]
        #5. Re-estimate the mean from aligned shapes.
        example =np.sum(data,axis=1)/14
        #6. Apply constraints on scale and orientation to the current estimate of the
        #mean by aligning it with x0 and scaling so that |x| = 1.
        '''
        meanx=0
        meany=0
        for j in range(0,80,2):
                meanx=meanx+example[j]
                meany=meany+example[j+1]
        for j in range(0,80,2):
                example[j]=example[j]-meanx/40
                example[j+1]=example[j+1]-meany/40
        example=example/np.linalg.norm(example)
        '''
        #4.1 |xj|=1
        example=example/np.linalg.norm(example)
        #4.2 find angle
        a=0
        b=0
        for j in range(0,80,2):
            a=example[j]*x0[j+1]-example[j+1]*x0[j]
            b=example[j]*x0[j]+example[j+1]*x0[j+1]
        angle=math.atan(a/b)
         #4.3 rotate
        for j in range(0,80,2):
            example[j]=math.cos(angle)*example[j]-math.sin(angle)*example[j+1]
            example[j+1]=math.sin(angle)*example[j]+math.cos(angle)*example[j+1]
        #7. If not converged, return to 4.
        if max(abs(example-examplestored))<0.01 :
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
    