import cv2
from pylab import imread
from skimage.color import rgb2gray

image = cv2.imread('Sample02/three-people.jpg')
cv2.imshow(' ',image)
cv2.waitKey(0)