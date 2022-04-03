from __future__ import print_function,unicode_literals,absolute_import,division
__author__ = 'MBI'
__doc__ = 'Script to geometric trasform of image'

#==== Packages ====#
import cv2
import numpy as np


#==== Classes ====#

class ImageGeometric():
    def __init__(self,image:np.uint8) -> None:
        self.image = image
        self.shape = image.shape[0:2]
    
    def imgInverted(self) -> np.uint8:
        img_dst = cv2.flip(self.image,1)
        return img_dst
    
    def imgRotated(self,value:int) -> np.uint8:
        rotation = cv2.getRotationMatrix2D((self.image.shape[0]/2.0,self.image.shape[1]/2.0),angle=value,scale=1)
        img_dst = cv2.warpAffine(self.image,rotation,dsize=self.shape)
        return img_dst


class ImageCartoon():
    def __init__(self,img:np.uint8) -> None:
        # @param sigma_s %Range between 0 to 200.
        # @param sigma_r %Range between 0 to 1.
        # @param shade_factor %Range between 0 to 0.1.
        self.image = img
        
    def getPencils(self,flag:int,sigS:int,sigR:float,shade_f:float) -> np.uint8:
        if (len(self.image.shape) == 3):
            pencil_0,pencil_1 = cv2.pencilSketch(self.image,sigma_s=sigS,sigma_r=sigR,shade_factor=shade_f)
        elif (len(self.image.shape) == 4):
            image = cv2.cvtColor(self.image,cv2.COLOR_BGRA2BGR)
            pencil_0,pencil_1 = cv2.pencilSketch(image,sigma_s=sigS,sigma_r=sigR,shade_factor=shade_f)
        else:
            image = cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
            pencil_0,pencil_1 = cv2.pencilSketch(image,sigma_s=sigS,sigma_r=sigR,shade_factor=shade_f)

        if (flag == 0): 
            return pencil_0
        else:
            return pencil_1
   
    def imageStyle(self,sigS:int,sigR:float) -> np.uint8:
        img_dst = cv2.stylization(self.image,sigS,sigR)
        return img_dst
    
    def combStyles(self,flag:int,sigS:int,sigR:float,shade_f:float) -> np.uint8:
        pencil_style = self.getPencils(flag,sigS,sigR,shade_f)
        img_dst = cv2.stylization(pencil_style,sigma_s=sigS,sigma_r=sigR)
        return img_dst


class ImageMorph():
    def __init__(self,img:np.uint8) -> None:
        # cv2.dilate
        # cv2.erode
        # cv2.morphologyEx
        self.img = img
        self.opedict:dict[str,object] = {
            "Open": cv2.MORPH_OPEN,
            "Close": cv2.MORPH_CLOSE,
            "Gradient": cv2.MORPH_GRADIENT,
            "TopHat": cv2.MORPH_TOPHAT,
            "BlackHat": cv2.MORPH_BLACKHAT,
            "Cross": cv2.MORPH_CROSS,
            "Ellipse": cv2.MORPH_ELLIPSE,
            "Dilate": cv2.MORPH_DILATE,
            "Erode": cv2.MORPH_ERODE
        }
    
    def imgDilate(self,kernel:int,iter:int) -> np.uint8:
        imgdilated = cv2.dilate(self.img,kernel=(kernel,kernel),iterations=iter)
        return imgdilated
    
    def imgErode(self,kernel:int,iter:int) -> np.uint8:
        imgerode = cv2.erode(self.img,kernel=(kernel,kernel),iterations=iter)
        return imgerode
    
    def imgMorphoEx(self,kernel:int,valuecv:str) -> np.uint8:
        imgmorpho = cv2.morphologyEx(self.img,op=self.opedict[valuecv],kernel=(kernel,kernel))
        return imgmorpho


    



#==== Main ====#
"""
imagePath = 'Image_Toolkit\\Cropped_20211106_124256.jpg'
image = cv2.imread(imagePath)
#imgTran = ImageGeometric(image)
imgTran = ImageCartoon(image)
#imgTran = ImageMorph(image)

#result = imgTran.imgRotated(90)
#result = imgTran.imgMorphoEx(7,"BlackHat")
result = imgTran.getPencils(0,20,0.5,0.10)

cv2.imshow('Img',image)
cv2.imshow('Morpho',result)

while True:
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
"""

