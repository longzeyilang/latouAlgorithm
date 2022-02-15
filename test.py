#coding:utf-8
import numpy 
import os
import subprocess
import shutil
import cv2
public_dir='/data/latoudatanew/images6_6am/'  
image_dir=public_dir+'2019_06_06/normal_latou'
algorithm_path='/home/gzy/latou/build/latou_algorithm'
temporal_path=public_dir+'temporal_latou/'
ng_path=public_dir+'ng/'
ok_path='/home/gzy/latou/images/ok/'
#data_path=ng_path   #单纯测试ng

#分别得到底座和拉片的图片
images=os.listdir(image_dir)
dizuo_image=[]
lapian_image=[]
for image in images:
     if image.find('dizuo')!=-1:  #找到底座
          dizuo_image.append(image)
     if image.find('lapian')!=-1: #找到拉片
          lapian_image.append(image)

n=0  #计数    
for index in range(len(dizuo_image)):
     dizuo=os.path.join(image_dir,dizuo_image[index])
     lapian=os.path.join(image_dir,lapian_image[index])
     model_calling_command=algorithm_path+' '+ temporal_path+' 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '+dizuo+' '+lapian
     print(model_calling_command)
     p=subprocess.Popen(model_calling_command,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True).communicate()[0].split('\n')
     # if 'dizuo res is: ng3' in p:    #排除无底座的拉片
     #      shutil.copy(dizuo,ng_path) 
     # if 'dizuo res is: ok' in p:     #排除无底座的拉片 
     #      shutil.copy(dizuo,ok_path)    
     # if 'blank res is: ok' in p:
     #      #print(dizuo)
     #      cv2.imshow("image",cv2.imread(dizuo))
     #      cv2.waitKey(0)
     #      cv2.destroyAllWindows()
     # continue

     if 'dizuo res is: ng' in p:
           n+=1
     #      cv2.imshow("image",cv2.imread(dizuo))
     #      cv2.waitKey(0)
     #      cv2.destroyAllWindows()
     if 'lapian res is: ng' in p:
           n+=1
     #      cv2.imshow("image",cv2.imread(lapian))
     #      cv2.waitKey(0)
     #      cv2.destroyAllWindows()
print(n)
