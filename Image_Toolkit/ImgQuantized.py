from __future__ import print_function,unicode_literals,absolute_import,division
__author__ = 'MBI'
__doc__ = 'Scrip to develop machine-learning on the image'

#==== Packages ====#
import numpy as np
import cv2

#==== Classes ====#

class Ecualized():
    def __init__(self,img:np.uint8) -> None:
        self.img = img
    
        self.criteria:dict[str,object] = {
            "EPS": cv2.TERM_CRITERIA_EPS,
            "MAX_ITER": cv2.TERM_CRITERIA_MAX_ITER,
            "EPS/MAX_ITER": cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        }
        self.flags:dict[str,object] = {
            "Random_Center": cv2.KMEANS_RANDOM_CENTERS,
            "PP_Center": cv2.KMEANS_RANDOM_CENTERS
        }

    
    def imgQuantization(self,k:int,att:int,critVal:str,flagVal:str,maxiter:int,epsilon:int) -> np.uint8:
        if (len(self.img.shape) == 3):
            imgfloat = np.float32(self.img).reshape(-1,3)
        elif (len(self.img.shape) == 4):
            imgfloat = np.float32(self.img).reshape(-1,4)
        else:
            imgfloat = np.float32(self.img).reshape(-1,2)

        criteria = (self.criteria[critVal],maxiter,epsilon)
        flags = self.flags[flagVal]

        _,label,centers = cv2.kmeans(imgfloat,k,None,criteria,att,
                                        flags)
        
        centers = np.uint8(centers)
        imgquantized = centers[label.flatten()]
        imgquantized = imgquantized.reshape(self.img.shape)
        return imgquantized
        

#==== Main ====#
"""
imagePath = 'Image_Toolkit\landscape_2.jpg'
image = cv2.imread(imagePath)

kimage = Ecualized(image)

result = kimage.imgQuantization(50,20,"EPS/MAX_ITER","PP_Center",70,1)


cv2.imshow('Img',image)
cv2.imshow('Ecualized',result)

while True:
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
"""
