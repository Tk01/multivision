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
 
 
def makeData(resize1 = 500, resize2 = 500, Upper=1):

    
    smallImages = np.zeros((14,resize1 * resize2))
    if Upper ==0 :
        listTopFourTeeth =[
        [(1339,1004),(1698,1261)],
        [(1346,1002),(1658,1223)],
        [(1357,1029),(1639,1245)],
        [(1383,1041),(1649,1290)],
        [(1361,991),(1669,1229)],
        [(1387,961),(1688,1210)],
        [(1368,1013),(1645,1251)],
        [(1357,887),(1610,1121)],
        [(1402,1065),(1633,1296)],
        [(1348,907),(1621,1175)],
        [(1316,1011),(1651,1290)],
        [(1394,985),(1658,1192)],
        [(1372,913),(1636,1195)],
        [(1344,1019),(1648,1325)]
        ]
    if Upper ==1 :
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
        [(x1,y1),(x2,y2)] = listTopFourTeeth[graphNumber-1]
        cutImage = img[y1:y2,x1:x2]
        result = cv2.resize(cutImage,(resize1,resize2), interpolation=cv2.INTER_NEAREST)
        smallImages[graphNumber-1] = result.flatten()
    return smallImages   
       
          
def findPosition(mean, eigenvectors, image, size1, size2,upper):
    m,n = image.shape
    
    best = 100000000
    bestKader = [(-1,-1),(-1,-1)]
    bestIm = np.zeros((500,400))
    for size11 in range(16,21,2):
        for size22 in range(11,21,3):
            s1 = int(size1 * size11 / 20)
            s2 = int(size2 * size22 / 20)
            if upper ==0 :
                a1 = int(m / 2 + m / 8)
                a2 = int(m / 2 + m / 4)
                aStep = 40
                b1 = int(n / 2 - n / 16)
                b2 = int(n / 2 + n / 16)
                bStep = 40
            if upper ==1 :
                a1 = int(m / 2 - m / 8)
                a2 = int(m / 2 + m / 8)
                aStep = 40
                b1 = int(n / 2 - n / 16)
                b2 = int(n / 2 + n / 16)
                bStep = 40
            
            for i in range(a1,a2,aStep):
                for j in range(b1,b2,bStep):
                    min1 = i - s2/2
                    max1 = i + s2/2
                    min2 = j - s1/2
                    max2 = j + s1/2
                    
                    cutImage = image[min1:max1,min2:max2]
                    
                    reCut = cv2.resize(cutImage,(size1,size2), interpolation=cv2.INTER_NEAREST)
                    X = reCut.flatten()
                    Y = project(eigenvectors, X, mean )
                    reX= reconstruct(eigenvectors, Y, mean)
                    
                    if np.linalg.norm(reX-X) < best:
                        best = np.linalg.norm(reX-X)
                        bestKader = [(min1,min2),(max1,max2)]
                        bestIm = reCut               
    return (bestKader,bestIm)


def findPositionFor(graphNumber, tooth):
    
    size1 = 500
    size2 = 400
    data = makeData(size1,size2)
    [eigenvalues, eigenvectors, mean] = pca(data,5)
    img = visualize.readRadiograph(graphNumber)
    isUpper = 0
    if tooth < 4:
        isUpper = 1
    
    [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2, isUpper)
    
    return [( b +(tooth-1)*(d-b)/4,a),(b +(tooth)*(d-b)/4,c)]

def findPositionForAll():
    allPositions = np.zeros((14,8,2,2))
    size1 = 500
    size2 = 400
    data = makeData(size1,size2,1)
    [eigenvalues, eigenvectors, mean] = pca(data,5)
    for i in range(1,15):
        img = visualize.readRadiograph(i)
        [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2,1)
        for j in range(1,5):
            allPositions[i-1,j-1] = [( b +(j-1)*(d-b)/4,a),(b +(j)*(d-b)/4,c)]
    size1 = 350
    size2 = 300
    data = makeData(size1,size2,0)
    [eigenvalues, eigenvectors, mean] = pca(data,5)
    for i in range(1,15):
        img = visualize.readRadiograph(i)
        [(a,b),(c,d)],im = findPosition(mean, eigenvectors, img, size1, size2,0)
        for j in range(1,5):
            allPositions[i-1,j+4-1] = [( b +(j-1)*(d-b)/4,a),(b +(j)*(d-b)/4,c)]
    return allPositions
    
