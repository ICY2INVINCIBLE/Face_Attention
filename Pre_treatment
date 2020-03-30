from mtcnn import MTCNN
import cv2
import time
import os
import csv
import pylab
import matplotlib.pyplot as plt
import dlib
from PIL import Image

#用了相对路径惨死，这里要用绝对路径
path="D:/pycharm_1/database/"
detector=MTCNN()
#mtcnn处理
num=1

def mtcnn(img,level):
    global num
    engagement,sum=read_csv(level)

    #print(engagement)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(imgRGB)
    #num=num+1
    for k,d in enumerate(faces):
        x,y,w,h=d["box"]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),1)
        #cv2.imshow('face', img)
        img=img[y:y+h,x:x+w]
        '''
        plt.figure()
        plt.imshow(img)
        pylab.show()
        '''
        break

    cv2.imwrite(path+'test/'+str(engagement)+"_"+str(sum)+"_" + str(num) +".jpg",img)
#    print(path + 'test_16300/' + str(engagement) + "_" + str(sum) + "_" + str(num) + '.jpg')
    num=num+1


def splitFrames(videoFileName,level):

    cap = cv2.VideoCapture(videoFileName)  # 打开视频文件
    num = 1
    times = 0
    print(videoFileName+"++++++++++++++++++++++++")
    while True:
        # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
        success, data = cap.read()
        if not success:
            break

        if (num % 21 == 0):
            mtcnn(data, level)
            #im.save('./' + str(num) + ".jpg")  # 保存当前帧的静态图像
            times = times + 1
        num = num + 1
        #print(num)
        if (times == 10):
            break


def read_csv(level):
    csv_path=path+"DAISEE/Labels/TestLabels.csv"
    csv_file=open(csv_path,'r')
    csv_reader_lines=csv.reader(csv_file)
    sum=1
    for line in csv_reader_lines:
        #print(line[2])
        if(sum==level):
            return line[2],level
        #line[2]是engagement的分数
        sum=sum+1

def change_size():
    size_path=path + "val_14290/"
    imgs=os.listdir(size_path)

    for img in os.listdir(size_path):
        img_path=size_path+img
        im=Image.open(img_path)
        out=im.resize((90,120))
        print(img_path+" "+str(out.size))
        #一定要save 不然变不了
        out.save(img_path)

#splitFrames(path+"DAISEE/DataSet/Train/100000plus/110001/1100011002/1100011002.avi")
def get_avi():
    root_path = path + "DAISEE/DataSet/Test"
    dir_file = os.listdir(root_path)
    total=1455
    for dir in dir_file:

        path2 = os.path.join(root_path, dir)  # path2=110001
        # print(path2)
        dir2 = os.listdir(path2)
        for files in dir2:
            path3 = os.path.join(path2, files)
            #print(path3)
            # 获得avi
            dir3 = os.listdir(path3)
            for file in dir3:
                path4=os.path.join(path3,file)
                #print(path4)
                total = total + 1
                splitFrames(path4, total)
                '''
                dir4=os.listdir(path4)
                for last_file in dir4:
                    last_path = os.path.join(path4, last_file)
                    #print(last_path+"         "+str(sum+1))
                    total=total+1
                    splitFrames(last_path, total)
                   '''
                    #print(last_path,total)
    print(total)

#以下两个函数是相互独立的，先运行get_avi，结束后再运行change_size()
#另外，get_avi()里面的循环最终是多少层取决于avi视频在文件的第几层，train的层数要比test和validation的要深一层
get_avi()
change_size()
