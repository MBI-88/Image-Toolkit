from __future__ import print_function,unicode_literals,absolute_import,division
__author__ = 'MBI'
__doc__ = 'Script to develop image filters'

#==== Packages ====#
import cv2
import numpy as np 

#==== Class ====#

class ImgFilter():
    def __init__(self,img:np.uint8) -> None:
        self.image = img
        
    def sobel(self,size:int,dx:int,dy:int,scale:float) -> np.uint8:
        # kise > order (dx,dy)
        # kernel 1,3,5,7
        # scale 0<-->1
        # dx o dy > 0
        image = cv2.Sobel(self.image,-1,dx,dy,(size,size),scale=scale)
        return image

    def scharr(self,dx:int,dy:int,scale:float) -> np.uint8:
        # dx >= 0 && dy >= 0 && dx+dy == 1
        # scale 0<-->1
        image = cv2.Scharr(self.image,-1,dx,dy,scale=scale)
        return image

    def laplacian(self,size:int,scale:float,delta:float) -> np.uint8:
        image = cv2.Laplacian(self.image,-1,(size,size),scale=scale,delta=delta)
        return image
    
    def bluring(self,size:int) -> np.uint8:
        # 0 <= anchor.x && anchor.x < ksize.width && 0 <= anchor.y && anchor.y < ksize.height
        image = cv2.blur(self.image,(size,size))
        return image
    
    def medianblur(self,size:int) -> np.uint8:
        image = cv2.medianBlur(self.image,size)
        return image
    
    def bilateralfilter(self,size:int,sigcolor:int,sigpace:int) -> np.uint8:
        # size = 5 default
        # sigcolor,sigspace <= 10 low effect | >= 150 stronger
        image = cv2.bilateralFilter(self.image,d=size,sigmaColor=sigcolor,sigmaSpace=sigpace)
        return image
    
    def filter2D(self,size:int,delta:float) -> np.uint8:
        image = cv2.filter2D(self.image,-1,(size,size),delta=delta)
        return image
    
    def gaussian(self,size:int,sigX:float,sigY:float) -> np.uint8:
        image = cv2.GaussianBlur(self.image,(size,size),sigX,sigY)
        return image
    
    

    
    

#==== Main ====#
"""
imgPath = 'Image_Toolkit\\landscape_2.jpg'
image = cv2.imread(imgPath)

imagecopy = image.copy()

imgfilter = ImgFilter(imagecopy)


imagefilter = imgfilter.scharr(1,0,1)

cv2.imshow('ImageFilter',imagefilter)
cv2.imshow('Image',image)

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
"""

