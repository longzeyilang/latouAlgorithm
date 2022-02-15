#coding:utf-8
import os
import psutil
import logging
import subprocess
import time
import ctypes
def get_pid(name):
    return map(int,subprocess.check_output(["pidof",name]).split())

def mark(pid):
    #设置一个日志输出文件
    log_filename="logging.txt"
    #设置日志输出格式
    log_format=' [%(asctime)s]  %(message)s'
    #日志文件基本设置
    logging.basicConfig(format=log_format,datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG,filename=log_filename,filemode='w')
    count=0
    while count<2000:
        p1=psutil.Process(pid)   #获取当前运行的pid
        a=str(p1.cpu_percent(interval=3))+",%.2f"%(p1.memory_percent())
        logging.info(a)
        count+=1

def load(file,cpu,mem):
    f=open(file,"r")
    fc=open(cpu,"w")
    fm=open(mem,"w")
    datas=f.readlines()
    for data in datas:
        da=data.split(" ")[4].split("\n")[0].split(",")
        fc.write(da[0]+'\n')
        fm.write(da[1]+'\n')
    f.close()
    fc.close()
    fm.close()


# if __name__ == "__main__":
#     pids=get_pid("latou_algorithm")
#     print(pids)
#     if pids is not None:
#         mark(pids[0])


if __name__ == "__main__":
    load("/home/gzy/latou/logging.txt","/home/gzy/latou/cpu.txt","/home/gzy/latou/mem.txt")
