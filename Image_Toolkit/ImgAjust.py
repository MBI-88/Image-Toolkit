from __future__ import print_function,unicode_literals,absolute_import,division
__author__ = 'MBI'
__doc__ = 'Backend to develop the Image App'
__version__ = '1.0.0'


#==== Package ====#
from PIL import ImageEnhance,Image
import cv2
import numpy as np



#==== Class ====#
class ImgAjust():
    def __init__(self,img:np.uint8) -> None:
        self.image = img
        
    def controlImage(self,valkey:dict[str,int]) -> np.uint8:
        image = Image.fromarray(self.image)
        
        # Bright
        brig_nor = float(valkey["Brightness"] / 100) 
        pilBrigh = ImageEnhance.Brightness(image)
        if (brig_nor == 0.5): 
            image = pilBrigh.enhance(1.0)
        else:
            if (brig_nor > 0.5): image = pilBrigh.enhance(brig_nor + 0.6)
            else:
                if (brig_nor == 0.0): image = pilBrigh.enhance(0.0)
                else:
                    image = pilBrigh.enhance(brig_nor + 0.40)
        
        # Contrast
        cont_nor = float(valkey["Contrast"] / 100)
        pilConstImg = ImageEnhance.Contrast(image)
        if (cont_nor == 0.50):
            image = pilConstImg.enhance(1.0)
        else: 
            if (cont_nor > 0.50): image = pilConstImg.enhance(cont_nor + 0.6)
            else: 
                if (cont_nor == 0.0) : image = pilConstImg.enhance(0.0)
                else: image = pilConstImg.enhance(cont_nor + 0.40)
        
        # Sharness
        shar_nor = float(valkey["Sharness"] / 100)
        pilSharnImg = ImageEnhance.Sharpness(image)
        if (shar_nor == 0.5):
            image = pilSharnImg.enhance(1.0)
        else:
            if (shar_nor > 0.5): 
                image = pilSharnImg.enhance(shar_nor + 0.6)
            else: 
                if (shar_nor == 0.0) : image = pilSharnImg.enhance(0.0)
                else: image = pilSharnImg.enhance(shar_nor + 0.40)

        # Color
        color_nor = float(valkey["Color"] / 100)
        pilColor = ImageEnhance.Color(image)
        if (color_nor == 0.5):
            image = pilColor.enhance(1.0)
        else:
            if (color_nor > 0.5): 
                image = pilColor.enhance(color_nor + 0.6)
            else:
                if (color_nor == 0.0): 
                    image = pilColor.enhance(0.0) 
                else: image = pilColor.enhance(color_nor + 0.40)
        
        # Tone/Saturarion
        tone_r = float(valkey["Tone"] / 100) # Hue 0 <-> 179
        satu_r = float(valkey["Saturation"] / 100) # Sa 0 <-> 255
        
        if (tone_r != 0.50 or satu_r != 0.50):
            image = np.array(image,dtype=np.uint8)
            if (len(image.shape) == 3):
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32)
            elif (len(image.shape) == 2):
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32)
            else:
                image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32)
                
            (h,s,v) = cv2.split(image)
            if (tone_r > 0.5): 
                h *= (1.0 - tone_r * 0.25) 
                image = cv2.merge((h,s,v)).astype(np.uint8)
                image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
            else: 
                h /= (tone_r + 0.35)
                image = cv2.merge((h,s,v)).astype(np.uint8)
                image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

            if (satu_r > 0.5):
                 s /= (1.0 - satu_r * 0.38)  
                 image = cv2.merge((h,s,v)).astype(np.uint8)
                 image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)          
            else: 
                s *= (satu_r - 0.0001)
                image = cv2.merge((h,s,v)).astype(np.uint8)
                image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

        image = np.array(image,dtype=np.uint8)
        return image
    
   