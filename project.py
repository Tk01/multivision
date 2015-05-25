import cv2
import cv2.cv as cv
import numpy as np
import Image
import math
import visualize
import estimateClick

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
        visualize.showReallignedData(data)
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
    visualize.showReallignedData(data)
    return data
    
    
def generateModel(vectors,model,estimate):
    Xt=0
    Yt=0
    angle=0
    s=0
    
    [l,t]=vectors.shape
    b=np.zeros(t)
    while(True):
       Xt_recorded=Xt
       Yt_recorded=Yt
       angle_recorded=angle
       s_recorded=s
       b_recorded=b
       x=model+np.dot(vectors,b) 
       meanx_est=0
       meany_est=0
       meanx_mod=0
       meany_mod=0
       a=0
       c=0
       for j in range(0,80,2):
            meanx_est=meanx_est+estimate[j]
            meany_est=meany_est+estimate[j+1]
            meanx_mod=meanx_mod+x[j]
            meany_mod=meany_mod+x[j+1]
            a=a+x[j]*model[j]+model[j+1]*x[j+1]
            c=c+x[j]*estimate[j+1]-x[j+1]*estimate[j]
       a=a/np.linalg.norm(x)
       c=c/np.linalg.norm(x)
       s=math.sqrt(a*a+c*c)
       angle=math.atan(c/a)
       Xt=meanx_mod-meanx_est
       Yt=meany_mod-meany_est
       y=np.zeros(80)
       for j in range(0,80,2):
           y[j]=math.cos(angle)*(y[j]-Xt)+math.sin(angle)*(y[j]-Xt)
           y[j+1] =math.sin(angle)*(y[j+1]-Yt)-math.cos(angle)*(y[j+1]-Yt)
       yx=0
       for j in range(0,80):
           yx = yx+ y[j]*model[j]
       y2 = y/(yx)
       b=np.dot(np.transpose(vectors),(y2-model))
       if (abs(Xt-Xt_recorded) <0.01) and (abs(Yt-Yt_recorded) <0.01) and (abs(s-s_recorded) <0.01) and (abs(angle-angle_recorded) <0.01) and (abs(max(b-b_recorded)) <0.01):
           break   
    return [x,Xt, Yt, s, angle, b]
    
    
def getTestData():
    return
    
    
def fit(data,model,mean):
    #per image
    for graphNumber in range(1,15):
        img = visualize.readRadiograph(graphNumber)
        
        #step1 ask estimate
        
        [(x1,y1),(x2,y2)] = estimateClick.askForEstimate(img)
        print str(x1) + "," + str(y1) + " - " + str(x2) + "," + str(y2)
        lengthx =x2-x1
        lengthy =y2-y1
        vectorizedEdgeData = findVectorizedEdgeData(img,(x1-lengthx,y1-lengthy),(x2+lengthx,y2+lengthy))
        visualize.displayVectorizedEdgeData(img, vectorizedEdgeData)
        genModel = adaptMean(mean,(x1,y1),(x2,y2))
        Xt =0
        Yt =0
        s =0 
        angle =0
        b =0  
    
        #step2 examine the region around each point around Xi to find a new point Xi'
        #gebaseerd op edge detection en distance
        while True:
            img2 = img.copy()
            visualize.addLandmarks(img2, genModel,False)
            img2=cv2.resize(img,(1000,500))
            img2=cv2.resize(img2,(1000,500))
            cv2.imshow('img_res',img2)
            cv2.waitKey(0) 
            genModel2 = list(genModel)
            Xt2 =Xt
            Yt2 =Yt
            s2 =s
            angle2 =angle
            b2 =b
            genModel = improve(genModel,vectorizedEdgeData)
            img2 = img.copy()
            visualize.addLandmarks(img2, genModel,False)
            img2=cv2.resize(img,(1000,500))
            img2=cv2.resize(img2,(1000,500))
            cv2.imshow('img_res',img2)
            cv2.waitKey(0) 
        #step3 update paramaters
            [genModel,Xt, Yt, s, angle, b]= generateModel(model,genModel,mean)
            
        #step4 check constraints
            if s>1.2 or Xt>x2 or Xt<x1 or Yt>y2 or Yt<y1:
                break
        #scaling mag, rotatie mag, beide niet te veel, weinig translation verandering tov estimate
        #b mag veranderen binnen gegeven grenzen
        #repeat from 2 until convergence
            if math.abs(max(genModel2-genModel)) <0.01 and math.abs(Xt2-Xt) <0.01 and math.abs(Yt2-Yt) <0.01 and math.abs(s2-s) <0.01 and math.abs(angle2-angle) <0.01 and math.abs(b2-b) <0.01:
                #visualize genModel
                
                break
    return
    
def adaptMean(mean,(x1,y1),(x2,y2)):
    mean = list(mean)
    lengthx = x2-x1
    lengthy=  y2-y1
    centerx= x1+lengthx/2
    centery= y1+lengthy/2
    meanx = np.zeros(40)
    meany = np.zeros(40)
    for x in range(0,80,2):
        meanx[x/2]=mean[x]
        meany[x/2]=mean[x+1]
    meanx=centerx + meanx*lengthx/(max(meanx)-min(meanx))
    meany=centery + meany*lengthy/(max(meany)-min(meany))
    for x in range(0,80,2):
        mean[x]=meanx[x/2]
        mean[x+1]=meany[x/2]
    return mean
    
def findVectorizedEdgeData(img,(x1,y1),(x2,y2)):
    filter_length = 7
    sigma = 2
    result = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)    
    
    edges = cv2.Canny(result, 10 , 25)

    array = []
    [M,N] = np.shape(edges)
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            if x>=0 and x< M and y>=0 and y< N and edges[x,y] != 0:
                array.append((x,y))
    return array

def improve(mean,vectorizedEdgeData):
    states = np.zeros((9,80))
    nedges = np.zeros((9,80))
    #initialisser met 9 nearestEdgePoints van 1ste punt? (mss sneller?)
    for l in range(0,80,2):
        for i in range(-1,2):
            for j in range(-1,2):
                copyS = np.copy(states)#copy voor for i ? (want nu verlies je 1 van de 9
                copyN = np.copy(nedges)# mogelijkheden doordat je 1 naar andere kopieert aan eind
                nPoint = nearestEdgePoint(mean[l]+i,mean[l+1]+j,vectorizedEdgeData)
                for k in range (0,9):
                    copyS[k,l]=mean[l]+i
                    copyS[k,l+1]=mean[l]+j
                    copyN[k,l]=nPoint[0]
                    copyN[k,l+1]=nPoint[1]
                minv= np.linalg.norm(copyS[0,:]-copyN[0,:])
                minState=copyS[0,:]
                minNEdge=copyN[0,:]
                for k in range (1,9):
                    if minv > np.linalg.norm(copyS[k,:]-copyN[k,:]):
                        minv= np.linalg.norm(copyS[k,:]-copyN[k,:])
                        minState=copyS[k,:]
                        minNEdge=copyN[k,:]
                states[3*(i+1)+j+1,:]=minState
                nedges[3*(i+1)+j+1,:]=minNEdge
    minv= np.linalg.norm(states[0,:]-nedges[0,:])
    minState=nedges[0,:]
    for k in range (1,9):
        if minv > np.linalg.norm(states[k,:]-nedges[k,:]):
                minv= np.linalg.norm(states[k,:]-nedges[k,:])
                minState=nedges[k,:]
    return minState

def nearestEdgePoint(x,y,vectors):
    minv=(x-vectors[0][0])*(x-vectors[0][0])+(y-vectors[0][1])*(y-vectors[0][1])
    minx =vectors[0][0]
    miny =vectors[0][1]
    for point in vectors:
        if minv>(x-point[0])*(x-point[0])+(y-point[1])*(y-point[1]):
           minv=(x-point[0])*(x-point[0])+(y-point[1])*(y-point[1])
           minx =point[0]
           miny =point[1]
    return (minx,miny)
            
def PCA(data, nb_components = 0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    dataTrans = np.transpose(data)
    [n,d] = dataTrans.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    
    mean = np.average(dataTrans,axis=0)
    
    XminMean = np.zeros((d,n))
    for i in range(0,n):
        XminMean[:,i] = dataTrans[i] - mean
    xtx = np.dot(np.transpose(XminMean),XminMean)
    
    values, vectors = np.linalg.eig(xtx)
    
    values,vectorList = (list(x) for x in zip(*sorted(zip(values, range(0,n)), key=lambda pair: pair[0], reverse=True)))
    
    eigenvalues = values[:nb_components]
    eigenvectorsTemp = np.zeros((n,nb_components))
    for i in range(0,nb_components):
        eigenvectorsTemp[:,i] = vectors[:,vectorList[i]]
    
    eigenvectors = np.zeros((d,nb_components))
    for i in range(0,nb_components):
        v = np.dot(XminMean,eigenvectorsTemp[:,i])
        eigenvectors[:,i] = v/np.linalg.norm(v)
    
    return [eigenvalues, eigenvectors, mean]
    
if __name__ == '__main__':
    data = getModelData()
    reallignedData = reallign(data)
    [values, vectors, mean] = PCA(reallignedData)
    testData = getTestData()
    result = fit(reallignedData, vectors, mean)
    