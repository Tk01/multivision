import cv2
import numpy as np
import math
import visualize
import initialPosition
numbersOfVectors = 10
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
def getModelData(tooth):
    result= np.zeros((80,14))
    for i in range(14):
        f = open('Landmarks\original\landmarks'+str(i+1)+'-'+str(tooth)+'.txt', 'r')
        t=0
        for j in f:
            result[t,i]=int(float(j.rstrip()))
            t = t+1
    return result
def reallign(data):
    for i in range(14):
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
        for i in range(14):
            data[:,i] = data[:,i]/np.linalg.norm(data[:,i])
            a=0
            b=0
            for j in range(0,80,2):
                a=data[j,i]*example[j+1]-data[j+1,i]*example[j]
                b=data[j,i]*example[j]+data[j+1,i]*example[j+1]
            angle=np.arctan(a/b)
            #4.3 rotate
            for j in range(0,80,2):
                data[j,i]=math.cos(angle)*data[j,i]-math.sin(angle)*data[j+1,i]
                data[j+1,i]=math.sin(angle)*data[j,i]+math.cos(angle)*data[j+1,i]
        example =np.sum(data,axis=1)/(14)
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
def fitOgr(dataList,vectorsList,meanList,sobelxList,sobelyList,p1):
    for graphNumber in range(1,15):
        finalModels = np.zeros((8,80))
        res=0
        sobelx = sobelxList[graphNumber-1]
        sobely = sobelyList[graphNumber-1]
        img = visualize.readRadiograph(graphNumber)
        img3 = img.copy()
        for tnum in range(1,9):
            vectors = vectorsList[tnum-1]
            mean = meanList[tnum-1]
            [(x1,y1),(x2,y2)] = p1[graphNumber-1][tnum-1]
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
                    finalModels[tnum-1] = genModel
                    break

                if max(genModel[::2]) > x2+lengthx or max(genModel[1::2]) > y2+0.2*lengthy or min(genModel[::2]) < x1-lengthx or min(genModel[1::2]) < y1-0.2*lengthy:
                    finalModels[tnum-1] = genModel
                    break
                if(counter == var):
                    counter =0
        
        showSegmentations(graphNumber, finalModels)
        
    return

def showSegmentations(graphNumber, eightresults):
    image = visualize.readRadiograph(graphNumber)
    m,n = image.shape
    segmentation = np.zeros((m,n), np.uint8)
    for contour in eightresults:
        minA = int(min(contour[0::2]))-10
        maxA = int(max(contour[0::2]))+10
        minB = int(min(contour[1::2]))-10
        maxB = int(max(contour[1::2]))+10
        lengthB = maxB - minB
        lengthA = maxA - minA
        white = np.ones((lengthB+1,lengthA+1), np.uint8)
        for i in range(0,80,2):
            pixels = line(int(contour[(i) % 80]),int(contour[(i+1) % 80]),int(contour[(i+2) % 80]),int(contour[(i+3) % 80]))
            for (a,b) in pixels:
                white[b-minB,a-minA] = 0
                segmentation[b,a] = 1
                

        
        thelist = [(0,0),(lengthB,0),(0,lengthA),(lengthB,lengthA)]
        while thelist:
            newList = []
            for (b,a) in thelist:
                if a >= 0 and b >= 0 and a < lengthA+1 and b < lengthB+1 and white[b,a] == 1:
                    white[b,a] = 0
                    newList.append((b-1,a))
                    newList.append((b+1,a))
                    newList.append((b,a-1))
                    newList.append((b,a+1))
            thelist = newList 

        white = white * 255
        segmentation[minB:(maxB+1),minA:(maxA+1)] = segmentation[minB:(maxB+1),minA:(maxA+1)] + white
    
    
    cv2.imwrite('Results/segments,' + str(graphNumber) + '.jpg' ,segmentation)  
      

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
    for l in range(0,80,2):
        copyS = np.copy(states)
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
    return  (res1+ res2)
    
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
    reallignedData = [0]*8
    for i in range(1,9):
        data = getModelData(i)
        reallignedData[i-1] = reallign(data)
    p2=initialPosition.findPositionForAll()
    sobelxx = [0]*14
    sobelyy = [0]*14
    for graphNumber in range(1,15):
        img = visualize.readRadiograph(graphNumber)
        sobelxx[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobelyy[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    values = [0] * 8
    vectors = [0] * 8
    mean = [0] * 8
    for a in range(1,9):
        [values[a-1], vectors[a-1], mean[a-1]] = PCA(reallignedData[a-1],numbersOfVectors)
    testData = getTestData()
    result = fitOgr(reallignedData, vectors, mean,sobelxx,sobelyy,p2)            