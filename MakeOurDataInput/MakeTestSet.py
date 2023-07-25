
import sys,re,pickle,os
import numpy as np
import pandas as pd 

# ! need to move test images out of common folder that contains everyone. 

# ---------------------------------------------------------------------------- #

# ! special chosen test images 

# z = """ 
# BWSSlide11.png                        
# BWSSlide63.png                        
# DownSlide12.png                       
# DownSlide60.png                       
# PWSSlide87.png                        
# RSTS1Slide3.png      
# 18495_01_Normalyoungchild.png         
# 47500_01_Normalyoungchild.png         
# 74948_01_Normalyoungchild.png         
# 9121_01_Normalyoungchild.png                           
# 22q11DSSlide150.png 
# 22q11DSSlide277.png
# 22q11DSSlide314.png
# BWSSlide10.png   
# BWSSlide17.png 
# BWSSlide31.png 
# BWSSlide64.png      
# CdLSSlide4.png
# CdLSSlide121.png                    
# CdLSSlide124.png                      
# DownSlide45.png                      
# KSSlide133.png    
# NSSlide6.png                      
# NSSlide29.png    
# NSSlide96.png                       
# PWSSlide44.png   
# PWSSlide13.png                     
# RSTS1Slide57.png         
# RSTS1Slide4.png                     
# UnaffectedSlide165.png
# UnaffectedSlide167.png
# UnaffectedSlide212.png
# UnaffectedSlide217.png
# UnaffectedSlide225.png
# UnaffectedSlide228.png
# UnaffectedSlide234.png
# UnaffectedSlide235.png
# UnaffectedSlide238.png
# UnaffectedSlide239.png
# WHSSlide13.png                       
# WSSlide185.png
# WSSlide226.png
# WSSlide267.png
# WSSlide316.png
# """.split()

# ! as of dec 12 2022
z = """ 
18495_01_Normalyoungchild.png 
22q11DSSlide150.png 
22q11DSSlide277.png
22q11DSSlide314.png
47500_01_Normalyoungchild.png
74948_01_Normalyoungchild.png 
9121_01_Normalyoungchild.png 
BWSSlide10.png  
BWSSlide11.png 
BWSSlide17.png 
BWSSlide31.png 
BWSSlide63.png 
BWSSlide64.png 
CdLSSlide121.png 
CdLSSlide124.png 
CdLSSlide4.png
DownSlide12.png 
DownSlide45.png 
DownSlide60.png 
KSSlide133.png 
NSSlide29.png 
NSSlide6.png 
NSSlide7.png 
NSSlide8.png
NSSlide96.png 
PWSSlide13.png 
PWSSlide44.png 
PWSSlide87.png 
RSTS1Slide3.png 
RSTS1Slide4.png
RSTS1Slide57.png
UnaffectedSlide165.png
UnaffectedSlide167.png
UnaffectedSlide212.png
UnaffectedSlide217.png
UnaffectedSlide225.png
UnaffectedSlide228.png
UnaffectedSlide234.png
UnaffectedSlide235.png
UnaffectedSlide238.png
UnaffectedSlide239.png
WHSSlide13.png 
WSSlide316.png
""".split()

z = [i.strip() for i in z]

# ---------------------------------------------------------------------------- #
# FairFace-aligned-60k-agegroup-06012021-AlignPix255
# FairFace-aligned-60k-agegroup-06012021-AlignPix255-RmbgPix255

# source_img_path = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-AlignPix255'
source_img_path = 'TrimImg_align_255pix'


os.chdir('/data/duongdb/ManyFaceConditions12012022')

test_img_path = 'TrimImg_align_255pix_testset'

if not os.path.exists (test_img_path) :
  os.mkdir( test_img_path )
  
# ---------------------------------------------------------------------------- #

for i in z: 
  this_img = os.path.join(source_img_path,i)
  if os.path.exists(this_img): 
    if 'TrimImg' in source_img_path: 
      os.system ('mv ' + this_img + ' ' + os.path.join(test_img_path,i)) 
    else: 
      os.system ('scp ' + this_img + ' ' + os.path.join(test_img_path,i)) # save the same copy in FairFace
  else: 
    print ('not able to test ',this_img)


# ---------------------------------------------------------------------------- #

# ! check test images are strictly unique from train images. 
x1 = set (os.listdir(source_img_path)) # this will be the train set 
x2 = set (os.listdir(test_img_path))

assert len ( x1.intersection(x2) ) == 0
assert len ( x2.intersection(x1) ) == 0


