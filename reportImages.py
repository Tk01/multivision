import cv2
import project
import visualize
import initialPosition
import numpy as np

def TwoBoxes():
    graphNumber = 1
    tooth = 1
    [(x11,y11),(x22,y22)] = initialPosition.findPositionFor(graphNumber, tooth)
    inner = [(int(x11),int(y11)),(int(x22),int(y22))]
    [(x1,y1),(x2,y2)] = inner
    
    lengthx =x2-x1
    lengthy =y2-y1
    
    outer = [(x1-lengthx,int(y1-0.2*lengthy)),(x2+lengthx,int(y2+0.2*lengthy))]
    
    img = visualize.readRadiograph(graphNumber)
    cv2.rectangle(img, inner[0], inner[1], (255, 255, 255), 2)
    cv2.rectangle(img, outer[0], outer[1], (255, 255, 255), 2)
    #img = cv2.resize(img,(1500,750), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('img' ,img)
    #cv2.waitKey(0)
    cv2.imwrite('Report/Images/twoBoxes.jpg',img)
    return

def PCAinitial():
    size1 = 500
    size2 = 400
    data = initialPosition.makeData(size1,size2,1)
    [eigenvalues, eigenvectors, mean] = initialPosition.pca(data,5)
    cv2.imwrite('Report/Images/PCAtop.jpg',np.hstack( (mean.reshape(size2,size1),
                                 initialPosition.normalize(eigenvectors[:,0].reshape(size2,size1)),
                                 initialPosition.normalize(eigenvectors[:,1].reshape(size2,size1)),
                                 initialPosition.normalize(eigenvectors[:,2].reshape(size2,size1)))
                               ).astype(np.uint8))
    #cv2.imshow('img',np.hstack( (mean.reshape(size2,size1),
    #                             initialPosition.normalize(eigenvectors[:,0].reshape(size2,size1)),
    #                             initialPosition.normalize(eigenvectors[:,1].reshape(size2,size1)),
    #                             initialPosition.normalize(eigenvectors[:,2].reshape(size2,size1)))
    #                           ).astype(np.uint8))
    #cv2.waitKey(0) 
        
    size1 = 350
    size2 = 300
    data = initialPosition.makeData(size1,size2,0)
    [eigenvalues, eigenvectors, mean] = initialPosition.pca(data,5)
    cv2.imwrite('Report/Images/PCAbottom.jpg',np.hstack( (mean.reshape(size2,size1),
                                 initialPosition.normalize(eigenvectors[:,0].reshape(size2,size1)),
                                 initialPosition.normalize(eigenvectors[:,1].reshape(size2,size1)),
                                 initialPosition.normalize(eigenvectors[:,2].reshape(size2,size1)))
                               ).astype(np.uint8))
    #cv2.imshow('img',np.hstack( (mean.reshape(size2,size1),
    #                             initialPosition.normalize(eigenvectors[:,0].reshape(size2,size1)),
    #                             initialPosition.normalize(eigenvectors[:,1].reshape(size2,size1)),
    #                             initialPosition.normalize(eigenvectors[:,2].reshape(size2,size1)))
    #                           ).astype(np.uint8))
    #cv2.waitKey(0) 
    
    
def initSearchSpace():
    graphNumber = 1
    
    img = visualize.readRadiograph(graphNumber)
    m,n = img.shape
    
    
    a1 = int(m / 2 - m / 8)
    a2 = int(m / 2 + m / 8)
    b1 = int(n / 2 - n / 16)
    b2 = int(n / 2 + n / 16)
    inner = [(b1,a1),(b2,a2)]
    
    a1 = int(m / 2 + m / 8)
    a2 = int(m / 2 + m / 4)
    b1 = int(n / 2 - n / 16)
    b2 = int(n / 2 + n / 16)
    outer = [(b1,a1),(b2,a2)]
    
    cv2.rectangle(img, inner[0], inner[1], (255, 255, 255), 2)
    cv2.rectangle(img, outer[0], outer[1], (255, 255, 255), 2)
    #img = cv2.resize(img,(1500,750), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('img' ,img)
    #cv2.waitKey(0)
    cv2.imwrite('Report/Images/searchSpace.jpg',img)

def divideInit():
    
    img = visualize.readRadiograph(1)
    
    posses = initialPosition.findPositionForAll()
    print posses[0]
    for [[aa,bb],[cc,dd]] in posses[0]:
        cv2.rectangle(img, (int(aa),int(bb)), (int(cc),int(dd)), (255, 255, 255), 2)
        
    #img = cv2.resize(img,(1500,750), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('img' ,img)
    #cv2.waitKey(0)
    cv2.imwrite('Report/Images/divide.jpg',img)

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
            pixels = project.line(int(contour[(i) % 80]),int(contour[(i+1) % 80]),int(contour[(i+2) % 80]),int(contour[(i+3) % 80]))
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
    
    
    cv2.imwrite('segment,' + str(graphNumber) + '.jpg' ,segmentation)
    
def testSegmentations():
    l = []
    l.append(project.getLandmarks(1))
    l.append(project.getLandmarks(2))
    l.append(project.getLandmarks(3))
    l.append(project.getLandmarks(4))
    l.append(project.getLandmarks(5))
    l.append(project.getLandmarks(6))
    l.append(project.getLandmarks(7))
    l.append(project.getLandmarks(8))

    showSegmentations(1,l)
    
def placeLandmarksOnSegments():
    for graphNumber in range(1,15):
        segment = cv2.imread('Results/segment,' + str(graphNumber) + '.jpg')
        
        for tooth in range(1,9):
            landmarks = project.getLandmarks(graphNumber, tooth)
            visualize.addLandmarks(segment, landmarks, color = (0,255,0))
    
        cv2.imwrite('Report/Images/segLand,' + str(graphNumber) + '.jpg', segment)

if __name__ == '__main__':
    #TwoBoxes()
    #PCAinitial()
    placeLandmarksOnSegments()
    print 1