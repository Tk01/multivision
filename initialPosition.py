import cv2
import numpy as np
import visualize
import estimateClick
import sys
    
def project(W, X, mu=None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu

def pca(X, num_components=0):
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
    
def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)
 
 
def makeData(resize1 = 500, resize2 = 500):
    
    results = np.zeros((14,resize2,resize1))
    
    smallImages = np.zeros((14,resize1 * resize2))
    
    listLeftTooth = [
    [(1305,743), (1415,1015)],
[(1307,626),(1403,965)],
[(1339,660),(1463,982)],
[(1322,679),(1420,1006)],
[(1345,733),(1478,984)],
[(1324,653),(1456, 930)],
[(1320,653),(1424,972)],
[(1357,588),(1478,881)],
[(1358,715),(1465,1053)],
[(1305,513),(1404,870)],
[(1251,647),(1374,967)],
[(1346,742),(1450,985)],
[(1322,532),(1430,846)],
[(1283,715),(1390,1008)] ]
    
    listTopFourTeeth = [
    [(1298,752),(1730,1017)],
[(1311,650),(1704,1004)],
[(1335,688),(1762,998)],
[(1318,684),(1697,1028)],
[(1346,728),(1758,993)],
[(1331,627),(1740,950)],
[(1318,660),(1707,993)],
[(1365,647),(1710,902)],
[(1365,749),(1737,1063)],
[(1307,519),(1669,876)],
[(1249,658),(1669,980)],
[(1342,753),(1692,998)],
[(1309,547),(1707,859)],
[(1281,709),(1680,1039)] ]

    for graphNumber in range(1,15):
        img = visualize.readRadiograph(graphNumber)
        
        #[(x1,y1),(x2,y2)] = estimateClick.askForEstimate(img)
        [(x1,y1),(x2,y2)] = listTopFourTeeth[graphNumber-1]
        cutImage = img[y1:y2,x1:x2]
        result = cv2.resize(cutImage,(resize1,resize2), interpolation=cv2.INTER_NEAREST)
        results[graphNumber-1,:,:] = result
        
        smallImages[graphNumber - 1] = result.flatten()
    
    allImages = np.zeros((resize2 * 2, resize1*8))
    for i in range(0,14):
        row = int(round(i / 8))
        col = i % 8
        allImages[resize2*row:resize2*(row+1),resize1*col:resize1*(col+1)] = results[i,:,:]
        #cv2.imshow('img' + str(i) ,np.uint8(results[i,:,:]))
        #cv2.waitKey(0)
    reAll = cv2.resize(allImages,(1500,750), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('all initialization images' ,np.uint8(reAll))
    #cv2.waitKey(0)
    
    return smallImages   
       
          
def findPosition(mean, eigenvectors, image, size1, size2):
    m,n = image.shape
    
    best = 100000000
    bestKader = [(-1,-1),(-1,-1)]
    bestIm = np.zeros((500,400))
    #try different sizes
    for size11 in range(16,21,2):
        for size22 in range(11,21,3):
            s1 = int(size1 * size11 / 20)
            s2 = int(size2 * size22 / 20)
            
            
            a1 = int(m / 2 - m / 8)
            a2 = int(m / 2 + m / 8)
            aStep = 40
            b1 = int(n / 2 - n / 16)
            b2 = int(n / 2 + n / 16)
            bStep = 40
            
            #loop over all possible starting positions
            for i in range(a1,a2,aStep):
                for j in range(b1,b2,bStep):
                    min1 = i - s2/2
                    max1 = i + s2/2
                    min2 = j - s1/2
                    max2 = j + s1/2
                    
                    cutImage = image[min1:max1,min2:max2]
                    
                    #cv2.imshow(str(size11)+ ' - ' + str(size22) + '  /  ' + str(i) + ' - ' + str(j) ,cutImage)
                    #cv2.waitKey(0)
                    
                    reCut = cv2.resize(cutImage,(size1,size2), interpolation=cv2.INTER_NEAREST)
                    X = reCut.flatten()
                    Y = project(eigenvectors, X, mean )
                    reX= reconstruct(eigenvectors, Y, mean)
                    
                    if np.linalg.norm(reX-X) < best:
                        #cv2.imshow(str(size11)+ ' - ' + str(size22) + '  /  ' + str(i) + ' - ' + str(j) ,reCut)
                        #cv2.waitKey(0)
                        best = np.linalg.norm(reX-X)
                        bestKader = [(min1,min2),(max1,max2)]
                        bestIm = reCut
                        #print best
                        sys.stdout.flush()
                        
            #print str(s1) + " - " + str(s2)
            sys.stdout.flush()                
    return (bestKader,bestIm)

def testMatch(mean, vectors):
    
    image = visualize.readRadiograph(1)
    
    
    [(x1,y1),(x2,y2)] = estimateClick.askForEstimate(image)
    
    cutImage = image[y1:y2,x1:x2]
                
    reCut = cv2.resize(cutImage,(size1,size2), interpolation=cv2.INTER_NEAREST)
    X = reCut.flatten()
    Y = project(eigenvectors, X, mean )
    reX= reconstruct(eigenvectors, Y, mean)
    
    
    print np.linalg.norm(reX-X)
    sys.stdout.flush()
    #cv2.imshow('img' ,reCut)
    #cv2.waitKey(0)


def findPositionFor(graphNumber, tooth):
    
    size1 = 500
    size2 = 400
    data = makeData(size1,size2)
    [eigenvalues, eigenvectors, mean] = pca(data,5)
    img = visualize.readRadiograph(graphNumber)
    [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2)
    
    return [( b +(tooth-1)*(d-b)/4,a),(b +(tooth)*(d-b)/4,c)]

def findPositionForAll():
    allPositions = np.zeros((14,4,2,2))
    size1 = 500
    size2 = 400
    for i in range(1,15):
        data = makeData(size1,size2)
        [eigenvalues, eigenvectors, mean] = pca(data,5)
        img = visualize.readRadiograph(i)
        [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2)
        for j in range(1,5):
            allPositions[i-1,j-1] = [( b +(j-1)*(d-b)/4,a),(b +(j)*(d-b)/4,c)]
    
    return allPositions
    
if __name__ == '__main__':
    size1 = 500
    size2 = 400
    data = makeData(size1,size2)
    [eigenvalues, eigenvectors, mean] =  pca(data,5)

    
    #testMatch(mean,eigenvectors)
    
    allImages = np.zeros((size2 * 3, size1*5))
    
    
    for graphNumber in range(1,2):
         img = visualize.readRadiograph(graphNumber)
         [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2)
         print str(a) + ',' + str(b) + " - " + str(c)+','+str(d)
         sys.stdout.flush()
         i = graphNumber - 1
         row = int(round(i / 5))
         col = i % 5
         allImages[size2*row:size2*(row+1),size1*col:size1*(col+1)] = im
    
    cv2.imwrite('/Results/allResultss.jpg',np.uint8(allImages))
    
    cv2.imshow('all results of findPosition' ,np.uint8(allImages))
    cv2.waitKey(0)