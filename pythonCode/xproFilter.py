import cv2
import sys
import numpy as np

#Read input image
image = cv2.imread("../assets/anish.jpg")

#create a copy of input image to work on
output = image.copy()

#split into channels
B, G, R = cv2.split(output)

#define vignette scale
vignetteScale = 6

#calculate the kernel size
k = np.min([output.shape[1], output.shape[0]])/vignetteScale

#create kernel to get the Halo effect
kernelX = cv2.getGaussianKernel(output.shape[1], k)
kernelY = cv2.getGaussianKernel(output.shape[0], k)
kernel = kernelY * kernelX.T

#normalize the kernel
mask = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

#apply halo effect to all the three channels of the image
B = B + B*mask
G = G + G*mask
R = R + R*mask

#merge back the channels
output = cv2.merge([B, G, R])

output = output /2

#limit the values between 0 and 255
output = np.clip(output, 0, 255)

#convert back to uint8
output = np.uint8(output)

#split the channels
B, G, R = cv2.split(output)


#Interpolation values
redValuesOriginal = np.array([0, 42, 105, 148, 185, 255])
redValues =         np.array([0, 28, 100, 165, 215, 255 ])
greenValuesOriginal = np.array([0, 40, 85, 125, 165, 212, 255])
greenValues =         np.array([0, 25, 75, 135, 185, 230, 255 ])
blueValuesOriginal = np.array([0, 40, 82, 125, 170, 225, 255 ])
blueValues =         np.array([0, 38, 90, 125, 160, 210, 222])

#create lookuptable
allValues = np.arange(0, 256)

#create lookup table for red channel
redLookuptable = np.interp(allValues, redValuesOriginal, redValues)
#apply the mapping for red channel
R = cv2.LUT(R, redLookuptable)

#create lookup table for green channel
greenLookuptable = np.interp(allValues, greenValuesOriginal, greenValues)
#apply the mapping for red channel
G = cv2.LUT(G, greenLookuptable)

#create lookup table for blue channel
blueLookuptable = np.interp(allValues, blueValuesOriginal, blueValues)
#apply the mapping for red channel
B = cv2.LUT(B, blueLookuptable)

#merge back the channels
output = cv2.merge([B, G, R])

#convert back to uint8
output = np.uint8(output)

#adjust contrast
#convert to YCrCb color space
output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)

#convert to float32
output = np.float32(output)

#split the channels
Y, Cr, Cb = cv2.split(output)

#scale the Y channel
Y = Y * 1.2

#limit the values between 0 and 255
Y = np.clip(Y, 0, 255)

#merge back the channels
output = cv2.merge([Y, Cr, Cb])

#convert back to uint8
output = np.uint8(output)

#convert back to BGR color space
output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

#create window to display images
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("xpro", cv2.WINDOW_AUTOSIZE)

#display images
cv2.imshow("image", image)
cv2.imshow("xpro", output)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()