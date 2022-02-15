#coding:utf-8
import numpy 
import os
import cv2
import subprocess
import psutil

public_dir="/data/latoudata/images5_17pm/"
algorithm_path='/home/gzy/latou/build/latou_algorithm'
temporal_path=public_dir+'temporal_latou/'
count=0
while count<3000:
        # train=algorithm_path+' '+ temporal_path+' 1 1 202 58 256 227 128 111 385 166'
        # p=subprocess.Popen(train,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        # print(p.pid)
        
        test=algorithm_path+' '+ temporal_path+' 0 1 202 58 256 227 128 111 385 166'
        p=subprocess.Popen(test,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True).communicate()
        print(p)
        count+=1

