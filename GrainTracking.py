import trackpy as tp
import pims

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pandas import DataFrame
import numpy as np
    
from scipy import ndimage
from skimage.util import img_as_uint
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_local, gaussian
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray, gray2rgb

import os


## Global file paths ##

filePath = "Run1-08-10-850f-4P-Mixed/USB-Unpolarized/"
imagePath = filePath + "*.jpg"

resultsFolder = filePath + "Results/"
if (not os.path.exists(resultsFolder)):
    os.makedirs(resultsFolder)


initF = pims.open(imagePath)

## Range of Grain radii to search for ##
hough_radii = np.arange(50,75, 2)

if __name__ == '__main__':
    __spec__  = None
    packingFracs = []

    logFile = open((filePath + "TrackingLog.txt"), 'w')
    
    @pims.pipeline
    def preprocess_image(img):
        """
        Apply image processing functions to return a binary image
        """
        # Get only one color channel from the image
        img = img[:,:,1]
        #img = img[:, 150:-150]
        # Apply thresholds
        adaptive_thresh = threshold_local(img,51)
        idx = img > adaptive_thresh
        idx2 = img < adaptive_thresh
        img[idx] = 0
        img[idx2] = 255
        img = ndimage.binary_erosion(img)

        #img = ndimage.binary_dilation(img)
        # Crop as needed [ymin:ymax, xmin:xmax]
        img = img[480:-635, :]
        return img_as_uint(img) 


    frames = preprocess_image(pims.open(imagePath))
    
    features = pd.DataFrame()
    
    for x in range(0,len(frames)):
        
        ## Prefoming a gaussian smoothing on the image, 
        ## might need to play around with sigma values.
        t1 = gaussian(frames[x], sigma = 3)

        ## Using canny algorithm to detect circle edges in image, 
        ## might need to play around with sigma values.
        edges = canny(t1, sigma=5)
        
        ## Creating a range of circle radii to try and fit to edges using hough algorithm.
        ## Relies on a good estimation of grain radii.
        hough_res = hough_circle(edges, hough_radii)
        
        ## Getting centers and radii of detected circles, 
        ## forcing a minimum seperation distance of lower bound of radii test range.
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                                   min_xdistance=np.amin(hough_radii), 
                                                   min_ydistance=np.amin(hough_radii))

        ## Creating an image to overlay detected circles on, 
        ## not nessicary for calculations just visual indication of whats being tracked.
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        # image = gray2rgb(frames[x])
        
        ## Itteration through each circle, overlaying it on the image and adding it to a 
        ## dataframe to be passed into trackpy eventually.
        for center_y, center_x, radius in zip(cy, cx, radii):
            # circy, circx = circle_perimeter(center_y, center_x, radius,
            #                                 shape=image.shape)
            # image[circy, circx] = (220, 20, 20)
            features = features.append([{'y': center_y,
                             'x': center_x,
                             'frame': (x),
                             },])
        print("Done with frame %i" %(x+1))
        ## Showing and saving the created plot.
        # ax.imshow(image, cmap=plt.cm.gray)
        # fname = "test" + str(x+1) + ".jpg"
        # plt.savefig(fname)
        # plt.clf()
        
    ## Calculate the area taken by the grains and output it to a logFile. 
    ## Current values don't seem correct. 
    for x in range(0, len(frames)):
        pNum, cNum = features[features['frame'] == x].shape
        packingFracs.append(np.average(radii)**2 * 4*np.pi * pNum)
        logFile.write("Total Grain area for frame: " + str(x+1) + "\n")
        logFile.write("In pixels: " + str(packingFracs[x]))
        logFile.write("\n")


    ## Link frames togother and create a image showing tracked grains.
    search_range = np.amin(hough_radii)
    t = tp.link(features, search_range, memory=1)
    tp.plot_traj(t, superimpose=initF[0])
    #plt.savefig(filePath + "Traj" + ".jpg")
    
    ## Create dataframe contating velocity information of each tracked grain.
    data = pd.DataFrame()
    for item in set(t.particle):
        sub = t[t.particle==item]
        dvx = np.diff(sub.x)
        dvy = np.diff(sub.y)
        for x, y, dx, dy, frame in zip(sub.x[:-1], sub.y[:-1], dvx, dvy, sub.frame[:-1],):
            data = data.append([{'dx': dx, 
                                 'dy': dy, 
                                 'x': x,
                                 'y': y,
                                 'frame': frame,
                                 'particle': item,
                                }]) 
            
    ## Create images of velocity field for tracked grains
    for i in range(0,len(frames)):
        d = data[data.frame==i]
        plt.imshow(initF[i][480:-635, :])
        plt.quiver(d.x, d.y, 
                   d.dx, -d.dy, pivot='middle', headwidth=4, headlength=6, color='red')
        plt.axis('off')
        title = resultsFolder + "vel" + str(i+1) + ".jpg"
        plt.savefig(title, dpi=400)
        plt.clf()
 
         
    logFile.close()