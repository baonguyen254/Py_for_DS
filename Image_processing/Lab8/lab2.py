import cv2
from pylab import imread
from skimage.color import rgb2gray

image = cv2.imread('/media/qbao/DATA/BaoBao/HK5/Image_processing/Lab8/Sample08/My_ID.png')
cv2.imshow(' ',image)
cv2.waitKey(0)