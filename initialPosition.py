import cv2
import numpy as np
import visualize
import estimateClick
import sys
    
def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    d,n = W.shape
    result = np.zeros(n)
    
    XminMu = X - mu
    
    for i in range(0,n):
        result[i] = np.dot(np.transpose(W[:,i]),XminMu)
        
    return result

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    
    resultImage = np.zeros(len(mu))

    
    for i in range(0,len(Y)):
        resultImage = resultImage + (Y[i] * W[:,i])
    
    resultImage = resultImage + mu
    
    return resultImage    
        
def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    
    mean = np.average(X,axis=0)
    
    XminMean = np.zeros((d,n))
    for i in range(0,n):
        XminMean[:,i] = X[i] - mean
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
    
def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)
 
 
def makeData(resize1 = 500, resize2 = 500):
    
    results = np.zeros((14,resize2,resize1))
    
    smallImages = np.zeros((14,resize1 * resize2))
    for graphNumber in range(1,15):
        img = visualize.readRadiograph(graphNumber)
        
        [(x1,y1),(x2,y2)] = estimateClick.askForEstimate(img)
        cutImage = img[y1:y2,x1:x2]
        result = cv2.resize(cutImage,(resize1,resize2), interpolation=cv2.INTER_NEAREST)
        results[graphNumber-1,:,:] = result
        
        smallImages[graphNumber - 1] = result.flatten()
    
    for i in range(0,14):
        cv2.imshow('img' + str(i) ,np.uint8(results[i,:,:]))
        cv2.waitKey(0)
    
    return smallImages   
       
          
def findPosition(mean, eigenvectors, image, size1, size2):
    m,n = image.shape
    
    best = 100000000
    bestPos = (-1,-1)
    #try different sizes
    for size11 in range(18,23,2):
        for size22 in range(11,21,3):
            s1 = int(size1 * size11 / 20)
            s2 = int(size2 * size22 / 20)
            
            
            a1 = int(m / 2 + m / 10)
            a2 = int(m / 2 + m / 4)
            aStep = 40
            b1 = int(n / 2 - n / 8)
            b2 = int(n / 2)
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
                        cv2.imshow(str(size11)+ ' - ' + str(size22) + '  /  ' + str(i) + ' - ' + str(j) ,reCut)
                        cv2.waitKey(0)
                        best = np.linalg.norm(reX-X)
                        print best
                        sys.stdout.flush()
                        
                        bestPos =  (i,j)
            print str(s1) + " - " + str(s2)
            sys.stdout.flush()                
    return bestPos

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
    cv2.imshow('img' ,reCut)
    cv2.waitKey(0)



if __name__ == '__main__':
    size1 = 100
    size2 = 400
    data = makeData(size1,size2)
    [eigenvalues, eigenvectors, mean] = pca(data)
    cv2.imshow('img',np.hstack( (mean.reshape(size2,size1),
                                 normalize(eigenvectors[:,0].reshape(size2,size1)),
                                 normalize(eigenvectors[:,1].reshape(size2,size1)),
                                 normalize(eigenvectors[:,2].reshape(size2,size1)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
    
    #testMatch(mean,eigenvectors)
    
    
    for graphNumber in range(1,15):
         img = visualize.readRadiograph(graphNumber)
         (a,b) = findPosition(mean, eigenvectors, img, size1, size2)
         print str(a) + " - " + str(b)
         sys.stdout.flush()