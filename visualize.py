import cv2
import cv2.cv as cv
import numpy as np
import project

def addLandmarks(img, landmarks, lines = True):
    
    for i in range(0,len(landmarks),2):
        x = int(landmarks[i])
        y = int(landmarks[i+1])
        cv2.circle(img,(x,y), 5, (255,255,255), -1)
    if lines:
        lenLand = len(landmarks)
        for i in range(0,lenLand,2):
            p1 = (int(landmarks[i]),int(landmarks[i+1]))
            p2 = (int(landmarks[(i+2)%lenLand]),int(landmarks[(i+3)%lenLand]))
            cv2.line(img,p1,p2, (255,255,255), 2)

def showLandmarksOnAllRadioGraphs(realIm = True, show = False):
    for n in range(1,15):
        showLandmarksOnRadioGraph(n,realIm, show)
        
        
#GraphNumber = welke RadioGraph je wil
#realIm = of je de foto als achtergrond wil of gwn een zwarte achtergrond
#show = je wil dat de tekening wordt getoond (past momenteel niet in het scherm)    
def showLandmarksOnRadioGraph(graphNumber, realIm = True, show = False):
    result= np.zeros((80))
    if realIm:
        if graphNumber < 9:
            img = cv2.imread("Radiographs/0"+str(graphNumber)+".tif",0)
        else:
            img = cv2.imread("Radiographs/"+str(graphNumber)+".tif",0)
    else:
        img = np.zeros((1603,3023,3), np.uint8)
    for i in range(8):
        f = open('Landmarks\original\landmarks'+str(graphNumber)+'-'+str(i+1)+'.txt', 'r')
        t=0
        for j in f:
            result[t]= int(float(j.rstrip()))
            t = t+1
        
        addLandmarks(img,result)
        
    cv2.imwrite('vis'+str(graphNumber)+'.jpg',img)
    
    if show:
        cv2.imshow('img',img)
        cv2.waitKey(0) 

def showReallignedData(data):
    windowSize = 500
    m,n = data.shape
    #maxX = 0
    #maxY = 0
    #for j in range(0,m,2):
    #    maxX = max(maxX,np.amax(abs(data[j,:])))
    #    maxY = max(maxY,np.amax(abs(data[j+1,:])))
    
    multiplier = (windowSize*0.45)/np.amax(abs(data))
    resizedData = data*int(multiplier)+windowSize/2
    
    img = np.zeros((windowSize,windowSize,3), np.uint8)
    for i in range(n):
        addLandmarks(img, resizedData[:,i], lines= True)
    
    cv2.imwrite('reallignedData.jpg',img)
    cv2.imshow('img',img)
    cv2.waitKey(0) 
    
if __name__ == '__main__':
    #showLandmarksOnAllRadioGraphs()
    #showLandmarksOnRadioGraph(1)
    data = project.getModelData()
    reallignedData = project.reallign(data)

    print reallignedData
    showReallignedData(reallignedData)