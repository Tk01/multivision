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
import initialPositionForTest as initialPosition
import random
numbersOfVectors = 6
iWeight1 = 1
iWeight2 = 1
lengthtraining = 5
lengthtest = 10
LeaveOneoutTest =1
Counter = 1
def lineData(tooth):
    data = getModelData(tooth)

    valuesLists = np.zeros((40,14,2*lengthtraining+1))
    
    for graphNumber in range(0,14-LeaveOneoutTest):
        img = visualize.readRadiograph((graphNumber+Counter)%14+1)
        toothData = data[:,graphNumber-1]        
        
        normals = getNormals(toothData)
        for i in range(0,80,2):
            lines = getLines((toothData[i],toothData[i+1]), (normals[i],normals[i+1]))
            valuesLists[i/2][graphNumber-1] = getValuesAtLine(img,lines)
            
    
    means = np.zeros((40,2*lengthtraining+1))
    covs = np.zeros((40,2*lengthtraining+1,2*lengthtraining+1))
    for i in range(0,40):
        means[i] = np.average(valuesLists[i], axis = 0)
        covs[i] = np.linalg.pinv(np.cov(valuesLists[i], rowvar = 0))
    return (means,covs)
    
def getLines((x,y),(nx,ny)):
    l1 = line(x,y,round(x + lengthtraining*nx),round(y + lengthtraining*ny))
    l2 = line(x,y,round(x - lengthtraining*nx),round(y - lengthtraining*ny))
    
    return (l1,l2)
    
def getValuesAtLine(img, lines):
    values = np.zeros(2 * lengthtraining + 1)
    
    (l1,l2) = lines
    
    for i in range(lengthtraining,-1,-1):
        values[lengthtraining-i] = img[l1[i][1],l1[i][0]]
    
    
    for i in range(1,lengthtraining+1):
        values[lengthtraining+i] = img[l2[i][1],l2[i][0]]
    if( np.count_nonzero(np.absolute(np.gradient(values))) == 0):
        return np.gradient(values)
    else: 
        return np.gradient(values)/np.sum(np.absolute(np.gradient(values)))
        
"http://rosettacode.org/wiki/Bitmap/Bresenham's_line_algorithm#Python"
def line(x0, y0, x1, y1):
    "Bresenham's line algorithm"
    array = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            array.append((x,y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            array.append((x,y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy   
    array.append((x,y))
    return array
def getLandmarks(tooth):
    result= np.zeros((80))
    f = open('Landmarks\original\landmarks'+str((14+Counter) % 14+1)+'-'+str(tooth)+'.txt', 'r')
    t=0
    for j in f:
        result[t]=int(float(j.rstrip()))
        t = t+1
    return result
def getModelData(tooth):
    result= np.zeros((80,14-LeaveOneoutTest))
    for i in range(14-LeaveOneoutTest):
        f = open('Landmarks\original\landmarks'+str((i+Counter) % 14+1)+'-'+str(tooth)+'.txt', 'r')
        t=0
        for j in f:
            result[t,i]=int(float(j.rstrip()))
            t = t+1
    return result
def reallign(data):
    for i in range(14-LeaveOneoutTest):
        x=0
        y=0
        for j in range(0,80,2):
            x=x+data[j,i]
            y=y+data[j+1,i]
        for j in range(0,80,2):
            data[j,i]=data[j,i]-x/40
            data[j+1,i]=data[j+1,i]-y/40
    example=data[:,0]/np.linalg.norm(data[:,0])
    x0=example
    while True:
        examplestored=example
        for i in range(14-LeaveOneoutTest):
            #4.1 |xj|=1
            data[:,i] = data[:,i]/np.linalg.norm(data[:,i])
            a=0
            b=0
            for j in range(0,80,2):
                a=data[j,i]*example[j+1]-data[j+1,i]*example[j]
                b=data[j,i]*example[j]+data[j+1,i]*example[j+1]
            angle=np.arctan(a/b)
            for j in range(0,80,2):
                data[j,i]=math.cos(angle)*data[j,i]-math.sin(angle)*data[j+1,i]
                data[j+1,i]=math.sin(angle)*data[j,i]+math.cos(angle)*data[j+1,i]
        example =np.sum(data,axis=1)/(14-LeaveOneoutTest)
        example=example/np.linalg.norm(example)
        a=0
        b=0
        for j in range(0,80,2):
            a=example[j]*x0[j+1]-example[j+1]*x0[j]
            b=example[j]*x0[j]+example[j+1]*x0[j+1]
        angle=np.arctan(a/b)
        for j in range(0,80,2):
            example[j]=math.cos(angle)*example[j]-math.sin(angle)*example[j+1]
            example[j+1]=math.sin(angle)*example[j]+math.cos(angle)*example[j+1]
        if max(abs(example-examplestored))<0.01 :
            break 
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
    xret=y+np.dot(P,b)
    for i in range(0,80):
        xret[i] =int(round(xret[i]))
    return xret  

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
    angle=np.arctan(c/a)
    return [Xt,Yt,s,angle]   
def getTestData():
    return
    
    
def fitDerivative(dataList,vectorsList,meanList,p1,teethData):
    global Counter
    for graphNumber in range(14,15):
        res=0
        img = visualize.readRadiograph((graphNumber+Counter) % 14 +1)
        img3 = img.copy()
        for tnum in range(1,9):
            vectors = vectorsList[tnum-1]
            mean = meanList[tnum-1]
            meanV = teethData[tnum-1][0]
            matrix = teethData[tnum-1][1]
            [(x1,y1),(x2,y2)] = p1[(graphNumber+Counter) % 14 ][tnum-1]
            lengthx =x2-x1
            lengthy =y2-y1
            genModel = adaptMean(mean,(x1,y1),(x2,y2))
            counter=0
            var = 50
            while True:
                if(counter==0 ):
                    genModelvar = list(genModel)
                    for i in range(0,80):
                        genModelvar[i]=int(round(genModelvar[i]))
                genModel2 = list(genModel)
                genModel = improve3(genModel,meanV,matrix,img)
                counter = counter +1
                genModel= generateModel2(vectors,mean,genModel)
                diff=0
                for i in range(0,80,2):
                    if not (genModel2[i] == genModel[i] and genModel2[i+1] == genModel[i+1]):
                            diff =diff+1
                if diff <5 or (counter == var and max(genModelvar - genModel)<10 and min(genModelvar - genModel)>-10):
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'succeeded'
                    break
                if max(genModel[::2]) > x2+lengthx or max(genModel[1::2]) > y2+0.2*lengthy or min(genModel[::2]) < x1-lengthx or min(genModel[1::2]) < y1-0.2*lengthy:
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'exceeded'
                    break
                if(counter == var):
                    counter =0
        img2=cv2.resize(img3,(1000,500))
        cv2.imwrite('Results/8teeth,der,' + str((graphNumber+Counter) % 14 +1) + ',' + str(numbersOfVectors) + ','+ str(res) + '.jpg',np.uint8(img2))
        print 'derivative,'+str((graphNumber+Counter) % 14 +1) +',' + str(res)
        
    return
def fitOgr(dataList,vectorsList,meanList,sobelxList,sobelyList,p1):
    global Counter
    for graphNumber in range(14,15):
        res=0
        sobelx = sobelxList[graphNumber-1]
        sobely = sobelyList[graphNumber-1]
        img = visualize.readRadiograph((graphNumber+Counter) % 14 +1)
        img3 = img.copy()
        for tnum in range(1,9):
            vectors = vectorsList[tnum-1]
            mean = meanList[tnum-1]
            [(x1,y1),(x2,y2)] = p1[(graphNumber+Counter) % 14 ][tnum-1]
            lengthx =x2-x1
            lengthy =y2-y1
            genModel = adaptMean(mean,(x1,y1),(x2,y2))
            counter=0
            var = 50
            while True:
                if(counter==0 ):
                    genModelvar = list(genModel)
                    for i in range(0,80):
                        genModelvar[i]=int(round(genModelvar[i]))
                genModel2 = list(genModel)
                genModel = improve(genModel,sobelx,sobely)
                counter = counter +1
                genModel= generateModel2(vectors,mean,genModel)
                diff=0
                for i in range(0,80,2):
                    if not (genModel2[i] == genModel[i] and genModel2[i+1] == genModel[i+1]):
                            diff =diff+1
                if diff <5 or (counter == var and max(genModelvar - genModel)<10 and min(genModelvar - genModel)>-10):
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'succeeded'
                    break
                if max(genModel[::2]) > x2+lengthx or max(genModel[1::2]) > y2+0.2*lengthy or min(genModel[::2]) < x1-lengthx or min(genModel[1::2]) < y1-0.2*lengthy:
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'exceeded'
                    break
                if(counter == var):
                    counter =0
        img2=cv2.resize(img3,(1000,500))
        cv2.imwrite('Results/8teeth,ogr,' + str((graphNumber+Counter) % 14 +1) + ',' + str(numbersOfVectors) + ','+ str(res) + '.jpg',np.uint8(img2))
        print 'Orientated Gradient,'+str((graphNumber+Counter) % 14 +1) +',' + str(res)
        
    return 
def fitNE(dataList,vectorsList,meanList,p1):
    global Counter
    for graphNumber in range(14,15):
        res=0
        img = visualize.readRadiograph((graphNumber+Counter) % 14 +1)
        img3 = img.copy()
        for tnum in range(1,9):
            vectors = vectorsList[tnum-1]
            mean = meanList[tnum-1]
            [(x1,y1),(x2,y2)] = p1[(graphNumber+Counter) % 14 ][tnum-1]
            lengthx =x2-x1
            lengthy =y2-y1
            [array,vectorizedEdgeData] = findVectorizedEdgeData(img,(x1-lengthx,y1-lengthy),(x2+lengthx,y2+lengthy),tnum)
            genModel = adaptMean(mean,(x1,y1),(x2,y2))
            counter=0
            var = 50
            while True:
                if(counter==0 ):
                    genModelvar = list(genModel)
                    for i in range(0,80):
                        genModelvar[i]=int(round(genModelvar[i]))
                genModel2 = list(genModel)
                genModel = improve2(genModel,vectorizedEdgeData,array)[0]
                counter = counter +1
                genModel= generateModel2(vectors,mean,genModel)
                diff=0
                for i in range(0,80,2):
                    if not (genModel2[i] == genModel[i] and genModel2[i+1] == genModel[i+1]):
                            diff =diff+1
                if diff <5 or (counter == var and max(genModelvar - genModel)<10 and min(genModelvar - genModel)>-10):
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'succeeded'
                    break
                if max(genModel[::2]) > x2+lengthx or max(genModel[1::2]) > y2+0.2*lengthy or min(genModel[::2]) < x1-lengthx or min(genModel[1::2]) < y1-0.2*lengthy:
                    visualize.addLandmarks(img3, genModel,False)
                    cv2.rectangle(img3, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 2)
                    res = res+np.linalg.norm(genModel-getLandmarks(tnum))
                    print 'exceeded'
                    break
                if(counter == var):
                    counter =0
        img2=cv2.resize(img3,(1000,500))
        cv2.imwrite('Results/8teeth,ne,' + str((graphNumber+Counter) % 14 +1) + ',' + str(numbersOfVectors) + ','+ str(res) + '.jpg',np.uint8(img2))
        print 'Nearest Edge,'+str((graphNumber+Counter) % 14 +1) +',' + str(res)
        
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
    
def findVectorizedEdgeData(img,(x1,y1),(x2,y2),toothnumber):
    filter_length = 5
    sigma = 1
    if toothnumber <5:
        result1 = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)  
        edges1 = cv2.Canny(np.uint8(result1), 15, 30)
    else:
        result1 = cv2.bilateralFilter(img,12,17,17)
        edges1 = cv2.Canny(np.uint8(result1), 1, 45)
    filter_length = 5
    sigma = 1
    if toothnumber <5:
        result = cv2.bilateralFilter(img,12,17,17)
        edges = cv2.Canny(np.uint8(result), 1, 45)
    else:
        result = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)  
        edges = cv2.Canny(np.uint8(result), 15, 30)  
    mid = (y1 + y2 ) / 2
    
    edges[0:mid][:] = edges1[0:mid][:]
    

    array = []
    [M,N] = np.shape(edges)
    for x in range(int(x1),int(x2+1)):
        for y in range(int(y1),int(y2+1)):
            if y>=0 and y< M and x>=0 and x< N and edges[y,x] != 0:
                array.append((y,x))
    return [array,scipy.spatial.KDTree(array)]
def improve3(estimate,mean,matrix,image):
    res = np.zeros(80);
    normals = getNormals(estimate)
    for i in range(0,80,2):
       li1= line(estimate[i],estimate[i+1],estimate[i]+lengthtest*normals[i],estimate[i+1]+lengthtest*normals[i+1])
       li2= line(estimate[i],estimate[i+1],estimate[i]-lengthtest*normals[i],estimate[i+1]-lengthtest*normals[i+1])
       bestV = value(image,mean[i/2],matrix[i/2],0,li1,li2)
       bestP = li1[0]
       for j in range(1,lengthtest-lengthtraining+1):
            if bestV>value(image,mean[i/2],matrix[i/2],j,li1,li2):
                bestV= value(image,mean[i/2],matrix[i/2],j,li1,li2)
                bestP = li1[j]
            if bestV>value(image,mean[i/2],matrix[i/2],-j,li1,li2):
                bestV= value(image,mean[i/2],matrix[i/2],-j,li1,li2)
                bestP = li2[j]
       res[i]=bestP[0]
       res[i+1]=bestP[1]          
    return res

def value(image,mean,matrix,j,li1,li2):
    line = np.zeros(2*lengthtraining+1);
    for i in range(-lengthtraining,lengthtraining+1):
        if j+i<0:
            line[i+lengthtraining] = image[li2[abs(j+i)][1],li2[abs(j+i)][0]]  
        else:
            line[i+lengthtraining] = image[li1[j+i][1],li1[j+i][0]]
    LineG =0
    if( np.count_nonzero(np.absolute(np.gradient(line))) == 0):
        LineG= np.gradient(line)
    else: 
        LineG= np.gradient(line)/np.sum(np.absolute(np.gradient(line)))
    return np.dot(np.dot(np.transpose(LineG-mean), matrix),LineG-mean)
        
def getNormals(estimate):
    res = np.zeros(80);
    for i in range(0,80,2):
        dx= estimate[(i+2) % 80]-estimate[(i-2) % 80]
        dy= estimate[(i+3) % 80]-estimate[(i-1) % 80]
        res[i] =-dy/max(abs(dy),abs(dx))
        res[i+1] = dx/max(abs(dy),abs(dx))
        if dy ==0 and dx ==0:
            res[i] =int(float(random.uniform(100, 200)))
            res[i+1] = int(float(random.uniform(100, 200)))
    return res
    
    
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
    
def improve(mean,sobelx,sobely):
    meanD=0
    hi=0
    for m in range(0,80,2):
        for n in range(m+2,80,2):
            meanD=meanD+distance((mean[m],mean[m+1]),(mean[n],mean[n+1]))
            hi=hi+1
    meanD=meanD/hi
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
    global iWeight1, iWeight2
    res1=0
    for i in range (2,l+1,2):
        res1=res1+(meanD-(copyS[i]-copyS[i-2])*(copyS[i]-copyS[i-2])+(copyS[i+1]-copyS[i-1])*(copyS[i+1]-copyS[i-1]))
    res2=0
    for i in range (4,l+1,2):
        res2=res2 + (copyS[i]-2*copyS[i-2]+copyS[i-4])*(copyS[i]-2*copyS[i-2]+copyS[i-4])+(copyS[i+1]-2*copyS[i-1]+copyS[i-3])*(copyS[i+1]-2*copyS[i-1]+copyS[i-3])
    return  (iWeight1*res1+iWeight2 * res2)
    
def extenerg(copyS,sobelx,sobely,l ):
    res=0
    for i in range(4,l,2):
        res=res-np.linalg.norm([sobelx[copyS[l-1]][copyS[l-2]],sobely[copyS[l-1]][copyS[l-2]]])*math.cos(np.arctan(sobely[copyS[l-1]][copyS[l-2]]/sobelx[copyS[l-1]][copyS[l-2]]) - (np.arctan((copyS[l+1]-copyS[l-3])/(copyS[l]-copyS[l-4]))-math.pi/2))
    return res
  

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
            
def PCA(X, num_components=0):
    X=np.transpose(X)
    [n,d] = X.shape
    if (num_components <= 0) or (num_components>n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]
    
if __name__ == '__main__':
    for g in range(0,14):
        Counter = g;
        reallignedData = [0]*8
        for i in range(1,9):
            data = getModelData(i)
            reallignedData[i-1] = reallign(data)
        teethdata =[0]*8
        for tnum in range(1,9):
            teethdata[tnum-1]=lineData(tnum)
            
        p2=initialPosition.findPositionForAll(LeaveOneoutTest,Counter)
        sobelxx = [0]*14
        sobelyy = [0]*14
        for graphNumber in range(1,15):
            img = visualize.readRadiograph(graphNumber)
            sobelxx[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
            sobelyy[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        for i in [4,6,8,10]:
            numbersOfVectors = i
            print str(numbersOfVectors)
            sys.stdout.flush()
            values = [0] * 8
            vectors = [0] * 8
            mean = [0] * 8
            for a in range(1,9):
                [values[a-1], vectors[a-1], mean[a-1]] = PCA(reallignedData[a-1],numbersOfVectors)
            testData = getTestData()
            result = fitNE(reallignedData, vectors, mean,p2)
            result = fitOgr(reallignedData, vectors, mean,sobelxx,sobelyy,p2)
            result = fitDerivative(reallignedData, vectors, mean,p2,teethdata)
            