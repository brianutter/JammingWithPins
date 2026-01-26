import pims
import matplotlib.pyplot as plt
import trackpy as tp
import pandas as pd
from pandas import DataFrame
import numpy as np

import time

## Global file paths ##
logFileName = "outputLog.txt"

imagePath = "Run1-07-27-1000f-WP-FW/W-Polarizer/*.jpg"


## Globals ## 
temp = []

#########################################################
#########################################################
#   Calcuates the G-Sqaured value on a image within a   #
#   region region. Logs output to file set above.       #
#    Normalizes to region given.                        #
#-------------------------------------------------------#
# Inputs:                                               #
#   Frame - ndarray containing the image                #
#   region - list containg the region to calculate over.#
#     First row is the x-bounds, second is y-bounds.    #
#                                                       #
# Returns:                                              #
#     Nothing                                           #
#                                                       #
#########################################################
#########################################################
def GSquaredOverRegion(frame, region):
    
    frame = frame.astype(np.float64)
        
    regionSum = 0 
    normalizeConst = (region[0][1] - 1) * (region[1][1] - 1)
    
    for x in range(region[0][0], region[0][1]):
        
        currSum = 0
        
        if (x == region[0][0]):
            x += 1
        if (x == region[0][1]):
            x -= 1
            
        for y in range(region[1][0], region[1][1]):
            
            if (y == region[1][0]):
                y -= 1
            if (y == region[1][1]):
                y += 1
            
            v = (frame[x+1][y] - frame[x-1][y])**2
            h = (frame[x][y+1] - frame[x][y-1])**2
            d1 = (frame[x+1][y+1] - frame[x-1][y-1])**2
            d1 /= 2
            d2 = (frame[x+1][y-1] - frame[x-1][y+1])**2
            d2 /= 2
            
            currSum = v + h + d1 + d2
            regionSum += currSum/normalizeConst
    
    s = str("Frame no#: " + str(frame.frame_no + 1) + "\n")
    logFile.write(s)
    s = str("Normalized G-Squared for region " + str(region) + " " + str(regionSum) + "\n")
    logFile.write(s)
    logFile.write("\n")

def GSquaredAtPix(frames, pLoc, nImage):
    
    xLim, yLim = frames[0].shape

    x = pLoc[0]
    y = pLoc[1]
    
    normalizeConst = 4
    
    if (x == xLim):
        x -= 1
    if (x == 0):
        x += 1
        
    if (y == yLim):
        y -= 1
    if (y == 0):
        y += 1
        
    currSum = 0
    totalSum = 0
    for frame in frames:
        frame = frame.astype(np.float64)

        v = (frame[x+1][y] - frame[x-1][y])**2
        h = (frame[x][y+1] - frame[x][y-1])**2
        d1 = (frame[x+1][y+1] - frame[x-1][y-1])**2
        d1 /= 2
        d2 = (frame[x+1][y-1] - frame[x-1][y+1])**2
        d2 /= 2

        currSum += v + h + d1 + d2
        
    
    nImage[x][y] = currSum / normalizeConst 

    return nImage
    # s = str("Frame no#: " + str(frame.frame_no + 1) + "\n")
    # logFile.write(s)
    # s = str("Normalized G-Squared for region " + str(region) + " " + str(regionSum) + "\n")
    # logFile.write(s)
    # logFile.write("\n")
#########################################################
#########################################################
#   Calcuates the G-Sqaured value over an image         #
#   Logs output to file set above                       #
#-------------------------------------------------------#
# Inputs:                                               #
#   Frame - ndarray containing the image                #
#                                                       #
# Returns:                                              #
#     Nothing                                           #
#                                                       #
#########################################################
#########################################################
def GSquaredOverFrame(frame):
    
    frame = frame.astype(np.float64)
    
    xmax, ymax = frame.shape
    
    imageSum = 0
    normalizeConst = (xmax - 1) * (ymax-1)
    
    initT = time.time()
    for x in range(1, xmax - 1):
        currSum = 0
        
        for y in range(1, ymax - 1):
            v = (frame[x][y+1] - frame[x][y-1])**2
            h = (frame[x+1][y] - frame[x-1][y])**2
            d1 = (frame[x+1][y+1] - frame[x-1][y-1])**2
            d1 /= 2
            d2 = (frame[x-1][y+1] - frame[x+1][y-1])**2
            d2 /= 2
            
            currSum = v + h + d1 + d2
            imageSum += currSum/normalizeConst
            
            
    finalT = time.time()
    
    ## Log output values
    s = str("Frame no#: " + str(frame.frame_no + 1) + "\n")
    logFile.write(s)
    s = str("Normalized G-Squared for frame: " + str(imageSum) + "\n")
    logFile.write(s)
    
    ## Log time taken to calculate
    s = str("Calculation Took " + str(int(finalT - initT)) + " Seconds" + "\n")
    logFile.write(s)
    logFile.write("\n")
    temp.append(imageSum)
    
#########################################################
#########################################################
#  Batch runs a G-Squared calculation over a collection #
#  of images.
#-------------------------------------------------------#
# Inputs:                                               #
#   Frame - ndarray containing the image                #
#                                                       #
# Returns:                                              #
#     Nothing                                           #
#                                                       #
#########################################################
#########################################################
def calcGSquared(frames):
    
    for x in range(0,len(frames)):
        
        GSquaredOverFrame(frames[x])
        print("Done with frame %i" %(x+1))
        
def calcAvgImage(frames, image):
    xlim, ylim = frames[0].shape
    
    for x in range(0, xlim-1):
        #print(x)
        for y in range(0, ylim-1):
            image = GSquaredAtPix(frames, [x,y], image)
            #print("x,y:", x,y)
            
    return image

if __name__ == '__main__':
    
    @pims.pipeline
    def cropSides(image):
        xmin = 320
        xmax = -290
        ymin = 1
        ymax = -1
        
        return image[ymin:ymax, xmin:xmax]
    
    @pims.pipeline
    def grey(image):
        return image[:,:,1]
    
    images = cropSides(pims.open(imagePath))
    
    logFile = open(logFileName, 'w')
    
    avgImage = np.zeros(images[0].shape)
    
    plt.imshow(images[0])
    
    initT = time.time()
    avgImage = calcAvgImage(images, avgImage)
    finalT = time.time()
    
    print(finalT - initT)
    
    plt.imshow(avgImage)
    
    #plt.imshow(images[0], cmap="gray")

    #calcGSquared(images)
    #GSquaredOverFrame(images[0])
    #GSquaredOverRegion(images[0], [[0,200], [200,250]])

    logFile.close()