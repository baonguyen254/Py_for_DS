import cv2
from pylab import imread
from skimage.color import rgb2gray

image = cv2.imread('Lab06 - Image/Skin 01.jpg')

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow(' ',image)
# cv2.imshow(' ',image_hsv)
cv2.waitKey(0)