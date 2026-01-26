from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import skimage
from skimage import io

from skimage.transform import rotate
import pims

imagePath = pims.ImageSequence('Polarized/*.jpg')
resultsFolder = "4Pin-polarized.ihm.jpg/"
if (not os.path.exists(resultsFolder)):
    os.makedirs(resultsFolder)

croppedPath4p = pims.ImageSequence('Results/Pol-Crop-4p/*.jpg')
croppedPath1p = pims.ImageSequence('Results/Pol-Crop-1p/*.jpg')
croppedPath2pAB = pims.ImageSequence('Results/Pol-Crop-2pAB/*.jpg')

croppedPath4p = pims.ImageSequence('Results/Pol-Crop-4p/*.jpg')
croppedPath1p = pims.ImageSequence('Results/Pol-Crop-1p/*.jpg')
croppedPath2pAB = pims.ImageSequence('Results/Pol-Crop-2pAB/*.jpg')

image = io.imread("Results/Pol-Crop-2pAB/Cropped2pinAB-1.jpg")
ax = plt.hist(image.ravel(), bins = 256)
plt.show()
image = io.imread("Results/Pol-Crop-4p/Cropped4pin-1.jpg")
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
training = io.imread("08-11/Run2-08-11-375f-4P-Mixed/FireWire-Polarized/Image1.jpg")
thres = 150
new = training > thres
training = training*new
plt.imshow(training, cmap=plt.cm.gray)
plt.show()
io.imsave("Results/training.jpg", training)

#55 for 4p
#150 option 1 for 2pAB

##Read an initial image from the dataset
img = io.imread("/work/Data/08-11 Unpol-1p/Image1.jpg")
##Rotate the image to get the experiment straight (no angle)
#trans = rotate(img, 1)
##Crop the image to remove apparatus from data|
##When cropping, the saved image may have wider margins, so may need to crop a little extra
trans = img[52:-50, :]

io.imsave("/work/Data/08-11 Unpol-1p/training.jpg", trans)
#Display for confirmation
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img)
ax[0].set_title('Original image')

ax[1].imshow(trans)
ax[1].set_title('Transformed')

#ax[2].imshow(img2, cmap=plt.cm.gray)
#ax[2].set_title('Cropped')

#To apply plasma filter, use cmap=plt.cm.plasma
#Can change coloring of heatmap by changing ending
#ax[3].imshow(img2, cmap=plt.cm.plasma)
#ax[3].set_title('Colored')



for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

a = 0
i = 1
thres = 155
while a < 172:
    training = io.imread("Results/Pol-Crop-1p/Cropped1pin-"+str(i)+".jpg")
    thres = 155
    new = training > thres
    training = training*new
    io.imsave("Results/Pol-Crop-1p-T/Cropped1pin-T-"+str(i)+".jpg", training)
    a = a + 1
    i = i + 1
#4pin run: -2 degrees, 350:-650, thres = 55
#1pin run: 1 degrees, 300:-570, thres = 155 (not very good)
#2pinAB run: 1 degrees, 300:-570, thres = 155 (not very good)

croppedPath4pT = pims.ImageSequence('Results/Pol-Crop-4p-T/*.jpg')
#Get that specific pixel of all the images (making a 2D array). Exclude non-existing images
vals4pT = np.array(croppedPath4pT);
avg4pT = np.mean(vals4pT, axis=0);

big = np.amax(croppedPath4pT)
print(big)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(avg4p, cmap=plt.cm.plasma)
ax[0].set_title('Original image')

ax[1].imshow(avg4pT, cmap=plt.cm.plasma)
ax[1].set_title('Transformed')
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

croppedPath2pABT = pims.ImageSequence('Results/Pol-Crop-2pAB-T/*.jpg')
#Get that specific pixel of all the images (making a 2D array). Exclude non-existing images
vals2pABT = np.array(croppedPath2pABT);
avg2pABT = np.mean(vals2pABT, axis=0);

croppedPath1pT = pims.ImageSequence('Results/Pol-Crop-1p-T/*.jpg')
#Get that specific pixel of all the images (making a 2D array). Exclude non-existing images
vals1pT = np.array(croppedPath1pT);
avg1pT = np.mean(vals1pT, axis=0);

croppedPath4pT = pims.ImageSequence('Results/Pol-Crop-4p-T-2/*.jpg')
#Get that specific pixel of all the images (making a 2D array). Exclude non-existing images
vals4pT = np.array(croppedPath4pT);
avg4pT = np.mean(vals4pT, axis=0);

big = np.amax(croppedPath2pABT)


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()


ax[0].imshow(avg2pABT, cmap=plt.cm.plasma)
ax[0].set_title('2 pin Transformed')

ax[1].imshow(avg1pT, cmap=plt.cm.plasma)
ax[1].set_title('1 pin Transformed')

ax[2].imshow(avg4pT, cmap=plt.cm.plasma)
ax[2].set_title('4 pin Transformed')
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

a = 0
i = 1
thres = 1
while a < 603:
    training = io.imread("08-11/Run2-08-11-603f-4P-Mixed/FireWire-Polarized/Image"+str(i)+".jpg")
    trans = rotate(training, 1)
    #Crop the image to remove apparatus from data
    #When cropping, the saved image may have wider margins, so may need to crop a little extra
    img2 = trans[:, 300:-570]
    io.imsave("Results/Pol-Crop-4p-3/Cropped4pin-3-"+str(i)+".jpg", img2)
    a = a + 1
    i = i + 1
#4pin run: -2 degrees, 350:-650, thres = 55
#1pin run: 1 degrees, 300:-570
#2pinAB run: 1 degrees, 300:-570, thres = 155

a = 0
i = 1
thres = 1
while a < 603:
    training = io.imread("Results/Pol-Crop-4p-3/Cropped4pin-3-"+str(i)+".jpg")
    thres = 155
    new = training > thres
    training = training*new
    io.imsave("Results/Pol-Crop-4p-T-3/Cropped4pin-T3-"+str(i)+".jpg", training)
    a = a + 1
    i = i + 1
#4pin run: -2 degrees, 350:-650, thres = 55
#1pin run: 1 degrees, 300:-570
#2pinAB run: 1 degrees, 300:-570, thres = 155