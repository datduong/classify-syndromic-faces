
# ! example 

import os
import cv2         
import numpy as np    
  
# path to input image is specified and
# image is loaded with imread command


from PIL import Image

# workdir = 'C:/Users/duongdb/Documents/ManyFaceConditions05092022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/EvalTestImgLabelIndex0/AverageAttr_test_Occlusion2.0'

# C:/Users/duongdb/Downloads/OneDrive_1_12-9-2022/14 original heatmaps
# 14 aggregate.png

# C:/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/EvalTestImgLabelIndex0/AverageAttr_test_Occlusion2.0
# 22q11DSSlide150v2_heatmappositiveAverage.png

workdir = 'C:/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/EvalTestImgLabelIndex0/AverageAttr_test_Occlusion2.0/'

os.chdir(workdir)


# image1 = Image.open('22q11DSSlide150v2_heatmappositiveAverage.png')
# new_image = Image.new("RGBA", (image1.size), "WHITE") # Create a white rgba background
# new_image.paste(image1, (0, 0), image1)              # Paste the image on the background. Go to the links given below for details.
# new_image.convert('RGB').save('test.jpg', "JPEG")  # Save as JPEG


imglist = sorted( os.listdir(workdir) )

imglist = [i for i in imglist if '22q11DSSlide150v2_heatmappositiveAverage.png' in i]

k = 50
for imgname in imglist : 

  image1 = cv2.imread(os.path.join(workdir,imgname))

  # cv2.cvtColor is applied over the
  # image input with applied parameters
  # to convert the image in grayscale
  img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

  img = cv2.boxFilter(img, -1, (k, k))
  # mask = 255 - gray_img

  # applying Otsu thresholding
  # as an extra flag in binary 
  # thresholding     
  # ret, thresh1 = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY + 
  #                                             cv2.THRESH_OTSU)     

  # ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
  ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)   

  # the window showing output image         
  # with the corresponding thresholding         
  # techniques applied to the input image    

  cv2.imwrite(os.path.join(workdir,'otsu_smooth_'+imgname), thresh1)



