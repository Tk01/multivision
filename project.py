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
import initialPosition
import random
#
# Landmarks zijn (x1,y1,x2,y2,...) geordend, zie visualize (heb normaal deze fout verbeterd)
#
numbersOfVectors = 6
iWeight1 = 1
iWeight2 = 1
lengthtraining = 5
lengthtest = 10
def lineData(tooth):
    data = getModelData(tooth)

    valuesLists = np.zeros((40,14,2*lengthtraining+1))
    
    for graphNumber in range(1,15):
        img = visualize.readRadiograph(graphNumber)
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
    #invert covariance
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
            if np.shape(array)[0] >10000:
                print 'large'
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            array.append((x,y))
            if np.shape(array)[0] >10000:
                print 'large'
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
        #visualize.showReallignedData(data)
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
            angle=np.arctan(a/b)
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
        angle=np.arctan(a/b)
         #4.3 rotate
        for j in range(0,80,2):
            example[j]=math.cos(angle)*example[j]-math.sin(angle)*example[j+1]
            example[j+1]=math.sin(angle)*example[j]+math.cos(angle)*example[j+1]
        #7. If not converged, return to 4.
        if max(abs(example-examplestored))<0.01 :
            break 
    #visualize.showReallignedData(data)
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
    #print b
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
    angle=np.arctan(c/a)
    return [Xt,Yt,s,angle]   
def getTestData():
    return
    
    
def fit(dataList,vectorsList,meanList,sobelxList,sobelyList,p1,teethData):
    #per image
    for graphNumber in range(1,15):
        #sobelx = sobelxList[graphNumber-1]
        #sobely = sobelyList[graphNumber-1]
        img = visualize.readRadiograph(graphNumber)
        img3 = img.copy()
        for tnum in range(1,9):
            #data = dataList[tnum-1]
            vectors = vectorsList[tnum-1]
            mean = meanList[tnum-1]
            meanV = teethData[tnum][0]
            matrix = teethData[tnum][1]
            #cv2.equalizeHist(img)
            #step1 ask estimate
            
            #[(x1,y1),(x2,y2)] = estimateClick.askForEstimate(img)
            [(x1,y1),(x2,y2)] = p1[graphNumber-1][tnum-1]
            #print str(x1) + "," + str(y1) + " - " + str(x2) + "," + str(y2)
            lengthx =x2-x1
            lengthy =y2-y1
            #[array,vectorizedEdgeData] = findVectorizedEdgeData(img,(x1-lengthx,y1-lengthy),(x2+lengthx,y2+lengthy))
            #visualize.displayVectorizedEdgeData(img, array)
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
                #Xt2 =Xt
                #Yt2 =Yt
                #s2 =s
                #angle2 =angle
                #b2 =b
                genModel = improve3(genModel,meanV,matrix,img)
                #sys.stdout.flush()
                counter = counter +1
                #print counter
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
                #print genModelvar - genModel
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
                    visualize.addLandmarks(img3, genModel,False)
                    print 'succeeded'
                    break
                #print str(x1-lengthx)+','+str(min(genModel[::2]))+','+str(max(genModel[::2]))+','+str(x2+lengthx)
                #print str(y1-lengthy)+','+str(min(genModel[1::2]))+','+str(max(genModel[1::2]))+','+str(y2+lengthy)
                if max(genModel[::2]) > x2+lengthx or max(genModel[1::2]) > y2+0.2*lengthy or min(genModel[::2]) < x1-lengthx or min(genModel[1::2]) < y1-0.2*lengthy:
                    visualize.addLandmarks(img3, genModel,False)
                    print 'exceeded'
                    break
                if(counter == var):
                    counter =0
        img2=cv2.resize(img3,(1000,500))
        #cv2.imshow('img_res4',img2)
        #cv2.waitKey(0) 
        cv2.imwrite('Results/8teeth,' + str(graphNumber) + ',' + str(numbersOfVectors) + ','+ str(iWeight1) + ','+ str(iWeight2) + '.jpg',np.uint8(img2))
        
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
    for i in range(2,l,2):
        res=res-np.linalg.norm([sobelx[copyS[l+1]][copyS[l]],sobely[copyS[l+1]][copyS[l]]])*math.cos(np.arctan(sobely[copyS[l+1]][copyS[l]]/sobelx[copyS[l+1]][copyS[l]]) - (np.arctan((copyS[l]-copyS[l-2])/(copyS[l+1]-copyS[l-1]))-math.pi/2))
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
    # or simply perform an economy size decomposition
    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]
    
if __name__ == '__main__':
    reallignedData = [0]*8
    for i in range(1,9):
        data = getModelData(i)
        reallignedData[i-1] = reallign(data)
    #list1 =[#(4,0.0,1.2),
    #    #(4,0.4,0.0),
    #    #(4,0.8,0.8),
    #    #(4,0.8,1.2),
    #    #(4,1.2,0.4),
    #    #(4,1.2,0.8),
    #    #(6,0.0,0.8),
    #    #(6,0.0,1.6),
    #    (6,0.4,0.4),
    #    (6,0.4,1.2),
    #    (6,0.8,0.4),
    #    (6,0.8,1.6),
    #    (6,1.2,0.0),
    #    (8,0.0,0.4),
    #    (8,0.0,1.2),
    #    (8,0.4,0.4),
    #    (8,0.4,0.8),
    #    (8,0.8,0.4),
    #    (8,0.8,0.8),
    #    (8,1.6,0.8),
    #    (8,1.6,1.2)]
    #kevin oneven numberOfVectors 1,3,5,..
    #tim even numberOfVectors 2,4,6,...
    teethdata =[0]*8
    for tnum in range(1,9):
        teethdata[tnum-1]=lineData(tnum)
        
    p2=initialPosition.findPositionForAll()
    sobelxx = [0]*14
    sobelyy = [0]*14
    #for graphNumber in range(1,15):
       # img = visualize.readRadiograph(graphNumber)
       # sobelxx[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
       # sobelyy[graphNumber-1] = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    #for (i,j,k) in list1:
    #    numbersOfVectors = i
    #    iWeight1 = j
    #    iWeight2 = k
    for i in [4,6,8,2,10,12,14]:
        numbersOfVectors = i
        #for j in range(0,20,4):
        #    iWeight1 = j/10.0
        #    for k in range(0,20,4):
        #        iWeight2 = k/10.0
        print str(numbersOfVectors)#+',' + str(iWeight1)+',' + str(iWeight2)
        sys.stdout.flush()
        values = [0] * 8
        vectors = [0] * 8
        mean = [0] * 8
        for a in range(1,9):
            [values[a-1], vectors[a-1], mean[a-1]] = PCA(reallignedData[a-1],numbersOfVectors)
        testData = getTestData()
        result = fit(reallignedData, vectors, mean,sobelxx,sobelyy,p2,teethdata)