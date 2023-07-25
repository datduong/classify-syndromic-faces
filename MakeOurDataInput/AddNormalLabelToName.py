
import os,sys,re,pickle

pathin = '/data/duongdb/ManyFaceConditions12012022/NormalImgChosenAsStressTest_align_255pix'

os.chdir(pathin)
for img in os.listdir(pathin): 
  os.system('mv ' + img + ' ' + 'Normal'+img )

  
