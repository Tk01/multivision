import cv2
import cv2.cv as cv
import numpy as np
import Image
def getModelData():
    return
def reallign(data):
    return
def generateModel(data):
    return    
def getTestData():
    return
def fit(data,model):
    return
def PCA(pcaData):
    return
if __name__ == '__main__':
    data = getModelData()
    reallignedData = reallign(data)
    pcaData = PCA(reallignedData)
    model = generateModel(pcaData)
    testData = getTestData()
    result = fit(testData,model)
    
        
        