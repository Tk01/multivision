import cv2
import cv2.cv as cv
import numpy as np
import Image
import math
import visualize
import estimateClick
import sys
import scipy.spatial
import scipy.signal
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
def generateModel2(P,mean,Y):
    [Xt,Yt,s,angle] =allign(mean,Y)
    y=np.zeros(80)
    for j in range(0,80,2):
        y[j] = math.cos(angle)*s*mean[j]-math.sin(angle)*mean[j+1]*s+Xt
        y[j+1] = math.sin(angle)*s*mean[j]+math.cos(angle)*mean[j+1]*s+Yt
    w=Y-y
    b=np.linalg.lstsq(P,w)[0]
    b=b/np.linalg.norm(b)*min(np.linalg.norm(b),100)
    print b
    #absb =0
    #for j in b:
    #    absb = absb +abs(j)
    #for i in range (1,14):
    #   b[i]= b[i]/abs(b[i])*min(abs(b[i]),i/14*np.sum(absb)) 
    xret=y+np.dot(P,b)
    for i in range(0,80):
        xret[i] =int(round(xret[i]))
    return xret  
    
def generateModel(P,mean,Y):
    Xt=0
    Yt=0
    angle=0
    s=1
    [l,t]=P.shape
    b=np.zeros(t)
    while(True):

       Xt_recorded=Xt
       Yt_recorded=Yt
       angle_recorded=angle
       s_recorded=s
       b_recorded=b
       
       x=mean +np.dot(P,b)
       [Xt,Yt,s,angle] =allign(x,Y)
       y=np.zeros(80)
       for j in range(0,80,2):
           y[j] = math.cos(angle)/s*(Y[j]-Xt)+math.sin(angle)*(Y[j+1]-Yt)/s
           y[j+1] = math.cos(angle)*(Y[j+1]-Yt)/s-math.sin(angle)*(Y[j]-Xt)/s
       yx=0
       for j in range(0,80):
           yx = yx+ y[j]*mean[j]
       y2 = y/(yx)
       b=np.dot(np.transpose(P),(y2-mean))
       if(max(abs(b-b_recorded))<0.01):
       #if (abs(Xt-Xt_recorded) <0.01) and (abs(Yt-Yt_recorded) <0.01) and (abs(s-s_recorded) <0.01) and (abs(angle-angle_recorded) <0.01):
        break
       #if (abs(Xt) <0.01) and (abs(Yt) <0.01) and (abs(s) <0.01) and (abs(angle) <0.01) and (abs(max(b)) <0.01):
       #    break
       
    
    for i in range(0,80,2):
        x1 = s*math.cos(angle)*x[i]-s*math.sin(angle)*(x[i+1])+Xt
        x2 = math.cos(angle)*(x[i+1])*s+math.sin(angle)*(x[i])*s+Yt
        x[i] = x1
        x[i+1] = x2

    return [x,Xt, Yt, s, angle, b]

def allign(x1,x2):
    meanx_est=0
    meany_est=0
    meanx_mod=0
    meany_mod=0
    a=0
    c=0
    v=0
    for j in range(0,80,2):
        meanx_est=meanx_est+x2[j]/40
        meany_est=meany_est+x2[j+1]/40
        meanx_mod=meanx_mod+x1[j]/40
        meany_mod=meany_mod+x1[j+1]/40
    Xt=meanx_est-meanx_mod
    Yt=meany_est-meany_mod
    for j in range(0,80,2): 
            a=a+(x1[j] - meanx_mod)*(x2[j]-meanx_est)+(x1[j+1]- meany_mod)*(x2[j+1]-meany_est)
            c=c+(x1[j] - meanx_mod)*(x2[j+1]-meany_est)-(x1[j+1] - meany_mod)*(x2[j]-meanx_est)
            v=v+(x1[j] - meanx_mod)*(x1[j] - meanx_mod)+(x1[j+1] - meany_mod)*(x1[j+1] - meany_mod)
    a=a/v
    c=c/v
    s=math.sqrt(a*a+c*c)
    angle=math.atan(c/a)
    return [Xt,Yt,s,angle]   
def getTestData():
    return
    
    
def fit(data,vectors,mean):
    #per image
    for graphNumber in range(2,15):
        img = visualize.readRadiograph(graphNumber)
        img3 = img#cv2.equalizeHist(img)
        #step1 ask estimate
        
        [(x1,y1),(x2,y2)] = estimateClick.askForEstimate(img)
        print str(x1) + "," + str(y1) + " - " + str(x2) + "," + str(y2)
        lengthx =x2-x1
        lengthy =y2-y1
        [array,vectorizedEdgeData] = findVectorizedEdgeData(img,(x1-lengthx,y1-lengthy),(x2+lengthx,y2+lengthy))
        visualize.displayVectorizedEdgeData(img, array)
        genModel = adaptMean(mean,(x1,y1),(x2,y2))
        Xt =0
        Yt =0
        s =1 
        angle =0
        [l,t]=vectors.shape
        b=np.zeros(t) 
        counter=0
        var = 50
        #step2 examine the region around each point around Xi to find a new point Xi'
        #gebaseerd op edge detection en distance
        while True:
            #counter = counter +1
            #if counter == 100:
            #    img2 = img.copy()
            #    visualize.addLandmarks(img2, genModel,False)
            #    img2=cv2.resize(img2,(1000,500))
            #    cv2.imshow('img_res1',img2)
            #    cv2.waitKey(0) 
            #    counter =0
            if(counter==0 ):
                genModelvar = list(genModel)
                for i in range(0,80):
                    genModelvar[i]=int(round(genModelvar[i]))
            genModel2 = list(genModel)
            Xt2 =Xt
            Yt2 =Yt
            s2 =s
            angle2 =angle
            b2 =b
            genModel = improve(img3,genModel)
            #sys.stdout.flush()
            counter = counter +1
            print counter
            #if counter == 500:
            #    img2 = img3.copy()
            #    visualize.addLandmarks(img2, genModel,True)
            #    img2=cv2.resize(img2,(1000,500))
            #    cv2.imshow('img_res2',img2)
            #    cv2.waitKey(0)
            #    counter =0
        #step3 update paramaters
            genModel= generateModel2(vectors,mean,genModel)
            #img2 = img.copy()
            #visualize.addLandmarks(img2, genModel,False)
            #img2=cv2.resize(img2,(1000,500))
            #cv2.imshow('img_res3',img2)
            #cv2.waitKey(0) 
        #step4 check constraints
            print genModelvar - genModel
        #scaling mag, rotatie mag, beide niet te veel, weinig translation verandering tov estimate
        #b mag veranderen binnen gegeven grenzen
        #repeat from 2 until convergence
            #if max(abs(genModel2-genModel)) <0.01 and abs(Xt2-Xt) <0.01 and abs(Yt2-Yt) <0.01 and abs(s2-s) <0.01 and abs(angle2-angle) <0.01 and max(abs(b2-b)) <0.01:
            diff=0
            for i in range(0,80,2):
                if not (genModel2[i] == genModel[i] and genModel2[i+1] == genModel[i+1]):
                        diff =diff+1
            if diff <5 or (counter == var and max(genModelvar - genModel)<10 and min(genModelvar - genModel)>-10):
                #visualize genModel
                img2 = img.copy()
                visualize.addLandmarks(img2, genModel,False)
                img2=cv2.resize(img2,(1000,500))
                cv2.imshow('img_res4',img2)
                cv2.waitKey(0) 
                break
            if(counter == var):
                counter =0
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
    meanx=centerx + meanx*lengthx/(max(meanx)-min(meanx))*0.75
    meany=centery + meany*lengthy/(max(meany)-min(meany))*0.75
    for x in range(0,80,2):
        mean[x]=meanx[x/2]
        mean[x+1]=meany[x/2]
    return mean
    
def findVectorizedEdgeData(img,(x1,y1),(x2,y2)):
    #bovenkant
    filter_length = 5
    sigma = 1
   # result = cv2.bilateralFilter(img,12,17,17)
    result1 = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)  
    edges1 = cv2.Canny(np.uint8(result1), 15, 30)
    #onderkant
    filter_length = 5
    sigma = 1
    result = cv2.bilateralFilter(img,12,17,17)
    edges = cv2.Canny(np.uint8(result), 1, 45)
    #result = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)  
    mid = (y1 + y2 ) / 2
    
    edges[0:mid][:] = edges1[0:mid][:]
    
    img2=cv2.resize(result,(1000,500))
    cv2.imshow('img_filtered',img2)
    cv2.waitKey(0)
    

    array = []
    [M,N] = np.shape(edges)
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            if y>=0 and y< M and x>=0 and x< N and edges[y,x] != 0:
                array.append((y,x))
    return [array,scipy.spatial.KDTree(array)]
    
def improve2(mean,vectorizedEdgeData,array):
    res = np.zeros(80);
    Nedges = np.zeros(80);
    for l in range(0,80,2):
        val = vectorizedEdgeData.query((mean[l+1],mean[l]))[0]
        result=(mean[l+1],mean[l])
        result2= vectorizedEdgeData.query((mean[l+1],mean[l]))[1]
        for i in range(-1,2):
            for j in range(-1,2):
                if(val > vectorizedEdgeData.query((mean[l+1]+i,mean[l]+j))[0]):
                    result=(mean[l+1]+i,mean[l]+j)
                    val=vectorizedEdgeData.query((mean[l+1]+i,mean[l]+j))[0]
                    result2 = vectorizedEdgeData.query((mean[l+1]+i,mean[l]+j))[1]
        res[l]=result[1]
        res[l+1]=result[0]
        Nedges[l]=array[result2][1]
        Nedges[l+1]=array[result2][0]
    return [res,Nedges]
    
def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    
def improve(img,mean):
    
    meanD=0
    for m in range(2,80,2):
        meanD=meanD=distance((mean[m],mean[m+1]),(mean[m-2],mean[m-1]))/39
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    states = np.zeros((9,80))
    #initialisser met 9 nearestEdgePoints van 1ste punt? (mss sneller?)
    for l in range(0,80,2):
        copyS = np.copy(states)#copy voor for i ? (want nu verlies je 1 van de 9
        for i in range(-1,2):
            for j in range(-1,2):
                
                copyS[:,l]=mean[l]+i
                copyS[:,l+1]=mean[l+1]+j                
                minv=intenerg(l,copyS[0],meanD)+ extenerg(copyS[0],sobelx,sobely,l )
                minState=copyS[0,:]
                for k in range (1,9):
                    if minv > intenerg(l,copyS[k],meanD)+ extenerg(copyS[k],sobelx,sobely,l ):
                        minv= intenerg(l,copyS[k],meanD)+ extenerg(copyS[k],sobelx,sobely,l )
                        minState=copyS[k,:]
                states[3*(i+1)+j+1,:]=minState
    minv= intenerg(78,copyS[0],meanD)+ extenerg(copyS[0],sobelx,sobely,78 )
    minState=states[0,:]
    for k in range (1,9):
        if minv > intenerg(78,copyS[k],meanD)+ extenerg(copyS[k],sobelx,sobely,78 ):
                minv= intenerg(78,copyS[k],meanD)+ extenerg(copyS[k],sobelx,sobely,78 )
                minState=states[k,:]
    return minState
def intenerg(l,copyS,meanD):
    
    res1=0
    for i in range (2,l+1,2):
        res1=res1+(meanD-(copyS[i]-copyS[i-2])*(copyS[i]-copyS[i-2])+(copyS[i+1]-copyS[i-1])*(copyS[i+1]-copyS[i-1]))
    res2=0
    for i in range (4,l+1,2):
        res2=res2 + (copyS[i]-2*copyS[i-2]+copyS[i-4])*(copyS[i]-2*copyS[i-2]+copyS[i-4])+(copyS[i+1]-2*copyS[i-1]+copyS[i-3])*(copyS[i+1]-2*copyS[i-1]+copyS[i-3])
    return  (0.5*res1+res2)
    
def extenerg(copyS,sobelx,sobely,l ):
    res=0
    for i in range(2,l,2):
        res=res-np.linalg.norm([sobelx[copyS[l]][copyS[l+1]],sobely[copyS[l]][copyS[l+1]]])*math.cos(math.atan(sobely[copyS[l]][copyS[l+1]]/sobelx[copyS[l]][copyS[l+1]]) - (math.atan((copyS[l+1]-copyS[l-1])/(copyS[l]-copyS[l-2]))+math.pi/2))
    return 5*res
  

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
    