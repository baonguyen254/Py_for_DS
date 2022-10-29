# 18110053_NguyenQuocBao
import numpy as np
import argparse
import math
import cv2

'''
*************************
*          Bài 1        *
*************************
'''
# TRANSLATION
def Translate(image, shift_distance):
    n,m = image.shape[:2]
    dx = shift_distance[0]
    dy = shift_distance[1]
    scan = np.array([[1,0,dx],[0,1,dy]])
    out_img = np.zeros(image.shape,dtype='u1')
    for i in range(n):
        for j in range(m):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x,origin_y,1])

            new_xy = np.dot(scan,origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < m and 0 < new_y < n:
                out_img[new_y,new_x] = image[i,j]
    return out_img

# flip
def flip(image, flipcode):
    # flip vertically
    if (flipcode == 0):
        n,m = image.shape[:2]
        dx = 0
        dy = n
        scan = np.array([[1,0,dx],[0,-1,dy]])
    # flip horizontally
    elif flipcode > 0:
        n,m = image.shape[:2]
        dx = m
        dy = 0
        scan = np.array([[-1,0,dx],[0,1,dy]])
    # flip vertically and horizontally
    elif flipcode < 0:
        n,m = image.shape[:2]
        dx = m
        dy = n
        scan = np.array([[-1,0,dx],[0,-1,dy]])
    out_img = np.zeros_like(image)
    for i in range(n):
        for j in range(m):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x,origin_y,1])

            new_xy = np.dot(scan,origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < m and 0 < new_y < n:
                out_img[new_y,new_x] = image[i,j]
    return out_img

# CROP
def Crop(image, x,y,height,width):
    crop_img = image[y:y+height, x:x+width]
    return crop_img

# Rotate
from PIL import Image

def Rotate(angle,x,y):
    '''
    |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) | 
    |0      1     |  |sin(𝜃)   1|  |0      1     |
    '''
    # shear 1
    tangent=math.tan(angle/2)
    new_x=round(x-y*tangent)
    new_y=y
    #shear 2
    new_y=round(new_x*math.sin(angle)+new_y)      #since there is no change in new_x according to the shear matrix
    #shear 3
    new_x=round(new_x-new_y*tangent)              #since there is no change in new_y according to the shear matrix
    return new_y,new_x

def resize(img, width,height):
    w, h = img.shape[:2]; 

  
    xNew = int(w * 1 / width)
    yNew = int(h * 1 / height) 

  
    xScale = xNew/(w-1) 
    yScale = yNew/(h-1) 

  
    newImage = np.zeros([xNew, yNew, 3], dtype = "u1")
    for i in range(xNew-1): 
        for j in range(yNew-1): 
            newImage[i + 1, j + 1]= img[1 + int(i / xScale), 1 + int(j / yScale)] 
    return newImage



#*******************************************************************************************

# TRANSLATE
image = cv2.imread("t_rex.jpg")
cv2.imshow("Original", image)
shift_distance = (100,100)
shift_img = Translate(image, shift_distance)
cv2.imshow("Down and right", shift_img)
image = cv2.imread("t_rex.jpg")
shift_distance = (-100,-100)
shift_img = Translate(image, shift_distance)
cv2.imshow("Up and left", shift_img)


#**********************************



# CROP
image = cv2.imread("t_rex.jpg")
cv2.imshow("Original", image)
y=0
x=0
h=200
w=200
crop_img = Crop(image,x,y,h,w)
cv2.imshow("cropped", crop_img)


#**********************************



# flip
# flipcode = 0: flip vertically
# flipcode > 0: flip horizontally
# flipcode < 0: flip vertically and horizontally
image = cv2.imread("t_rex.jpg")
cv2.imshow("Original", image)
flip_img1 = flip(image,0)
cv2.imshow("flipped vertically", flip_img1)
flip_img2 = flip(image,1)
cv2.imshow("flipped horizontally", flip_img2)
flip_img3 = flip(image,-1)
cv2.imshow("flipped vertically and horizontally", flip_img3)


#**********************************




# Rotate

image = cv2.imread("t_rex.jpg")             # Load the image
angle = -90
# angle=-int(input("Enter the angle :- "))                # Ask the user to enter the angle of rotation
# Define the most occuring variables
angle=math.radians(angle)                               #converting degrees to radians
cosine=math.cos(angle)
sine=math.sin(angle)
height=image.shape[0]                                   #define the height of the image
width=image.shape[1]                                    #define the width of the image
# Define the height and width of the new image that is to be formed
new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1
# define another image variable of dimensions of new_height and new _column filled with zeros
output=np.zeros((new_height,new_width,image.shape[2]))
image_copy=output.copy()

# Find the centre of the image about which we have to rotate the image
original_centre_height   = round(((image.shape[0]+1)/2)-1)    #with respect to the original image
original_centre_width    = round(((image.shape[1]+1)/2)-1)    #with respect to the original image
# Find the centre of the new image that will be obtained
new_centre_height= round(((new_height+1)/2)-1)        #with respect to the new image
new_centre_width= round(((new_width+1)/2)-1)          #with respect to the new image

for i in range(height):
    for j in range(width):
        #co-ordinates of pixel with respect to the centre of original image
        y=image.shape[0]-1-i-original_centre_height                   
        x=image.shape[1]-1-j-original_centre_width 
        #Applying shear Transformation                     
        new_y,new_x = Rotate(angle,x,y)
        '''since image will be rotated the centre will change too, 
            so to adust to that we will need to change new_x and new_y with respect to the new centre'''
        new_y=new_centre_height-new_y
        new_x=new_centre_width-new_x
        output[new_y,new_x,:]=image[i,j,:]                          #writing the pixels to the new destination in the output image

pil_img=Image.fromarray((output).astype(np.uint8))                # converting array to image
pil_img.show("rotated " ,output)




#**********************************


# Resize 
img = cv2.imread('t_rex.jpg') 
newImage = resize(img,2,2)
cv2.imshow("resize",newImage)


#**********************************


img = cv2.imread('t_rex.jpg') 
w, h = 199,130
img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
cv2.imshow("Resize using cv2",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
*************************
*          Bài 2        *
*************************
'''
def convolute(image,kernel):
    img_width = image.shape[1]
    img_height = image.shape[0]
    ker_width = kernel.shape[1]
    ker_height = kernel.shape[0]
    H = (ker_height - 1)//2
    W = (ker_width - 1)//2

    out = np.zeros_like(image)
    
    for i in range(H, img_height - H):
        for j in range(W, img_width - W):
            out[i - H, j - W] = np.tensordot(image[i - H:i + H + 1, j - W:j + W + 1], kernel, axes=((0, 1), (0, 1)))

    return out

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = False, help = "D:\BaoBao\HK5\Processing_statistics\tuan5")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
# cv2.imshow("Original", image)
# cv2.waitKey(0)


image = cv2.imread("t_rex.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# kernel = np.array([[1, 0, 0],
#                    [0, 1, 1],
#                    [1, 0, 1]])
# idenity = np.array([[0, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 0]])
Gaussian_3 = 1/16*np.array([[1, 2, 1],
                   [2, 4, 1],
                   [1, 2, 1]])
Gaussian_5= 1/273*np.array([[1, 4, 7, 4, 1],
                   [4, 16, 26, 16, 4],
                   [7, 26, 41, 26, 7],
                   [4, 16, 26, 16, 4],
                   [1, 4, 7, 4, 1]])
Averaging_5 = 1/25*np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])
Averaging_3 = 1/9*np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
blur_1 = convolute(image,Averaging_3)
blur_2 = convolute(image,Averaging_5)
blur_3 = convolute(image,Gaussian_3)
blur_4 = convolute(image,Gaussian_5)

# blur = convolute(image,kernel)
# cv2.imshow("blurred", blur)
cv2.imshow("blurred Averaging 3x3", blur_1)
cv2.imshow("blurred Averaging 5x5", blur_2)
cv2.imshow("blurred Gaussian 3x3", blur_3)
cv2.imshow("blurred Gaussian 5x5", blur_4)

cv2.waitKey(0)