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
        
def readRadiograph(graphNumber):
        if graphNumber < 10:
            img = cv2.imread("Radiographs/0"+str(graphNumber)+".tif",0)
        else:
            img = cv2.imread("Radiographs/"+str(graphNumber)+".tif",0)
            
        return img
        
        
#GraphNumber = welke RadioGraph je wil
#realIm = of je de foto als achtergrond wil of gwn een zwarte achtergrond
#show = je wil dat de tekening wordt getoond (past momenteel niet in het scherm)    
def showLandmarksOnRadioGraph(graphNumber, realIm = True, show = False):
    result= np.zeros((80))
    if realIm:
        img = readRadiograph(graphNumber)
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
        addLandmarks(img, resizedData[:,i], lines= False)
    
    cv2.imwrite('reallignedData.jpg',img)
    cv2.imshow('img',img)
    cv2.waitKey(0) 

#Toont de verandering in de vorm bij toevoegen van de eigenvectoren aan de mean
#figuur gerangschikt als: mean-x | mean | mean+x  met x = eigenvector * 0.1
#
#mean = de gemiddelde waarde gevonden door PCA
#eigenvectors = de gevonden eigenvectors, gerangschikt op grootte van eigenwaarde
#hoeveel verschillende eigenvectoren je wil zien
def showPCAdata(mean, eigenvectors, nb=3):
    windowSize = 500
    m,n = eigenvectors.shape
    
    multiplier = (windowSize*0.45)/np.amax(abs(mean))
    resizedMean = mean*int(multiplier)+windowSize/2
    
    imgMean = np.zeros((windowSize,windowSize,3), np.uint8)
    addLandmarks(imgMean, resizedMean, lines= True)
    
    for i in range(nb):
        imgEig1 = np.zeros((windowSize,windowSize,3), np.uint8)
        eig1 = mean + (eigenvectors[:,i] * -0.1)
        resizedEig1 = eig1*int(multiplier)+windowSize/2
        addLandmarks(imgEig1,resizedEig1, lines= True)
        
        imgEig2 = np.zeros((windowSize,windowSize,3), np.uint8)
        eig2 = mean + (eigenvectors[:,i] * 0.1)
        resizedEig2 = eig2*int(multiplier)+windowSize/2
        addLandmarks(imgEig2,resizedEig2, lines= True)
        
        cv2.imshow('verandering bij eigenvector '+str(i+1),np.hstack( ( imgEig1,
                                    imgMean,
                                    imgEig2 )
                                ).astype(np.uint8))
        cv2.waitKey(0) 

def showEdges():
    graphNumber = 1
    if graphNumber < 9:
        img = cv2.imread("Radiographs/0"+str(graphNumber)+".tif",0)
    else:
        img = cv2.imread("Radiographs/"+str(graphNumber)+".tif",0)
    
    #equ = cv2.equalizeHist(img)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #equ = clahe.apply(img)
    filter_length = 7
    sigma = 2
    result = cv2.GaussianBlur(img, (filter_length,filter_length),sigma)    
    
    edges = cv2.Canny(result, 10 , 25)

    canny_result = np.copy(result)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('img',canny_result)
    cv2.waitKey(0)
    
    cv2.imwrite('edgesData.jpg',canny_result)


    
    canny_empty = np.zeros((1603,3023,3), np.uint8)
    canny_empty[edges.astype(np.bool)]=255
    
    cv2.imwrite('edgesData2.jpg',canny_empty)


def displayVectorizedEdgeData(image, vectorizedEdgeData):
    m,n = image.shape
    img = np.zeros((m,n,3), np.uint8)
    
    for (x,y) in vectorizedEdgeData:
        img[x,y] = (255,255,255)
    
    img2=cv2.resize(img,(1000,500))
    cv2.imshow('img_res',img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    #showLandmarksOnAllRadioGraphs()
    #showLandmarksOnRadioGraph(1)
    #data = project.getModelData()
    #reallignedData = project.reallign(data)
    #[values, vectors, mean] = project.PCA(reallignedData)
    #showPCAdata(mean,vectors)
    showEdges()