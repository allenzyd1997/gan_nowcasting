import numpy as np
import cv2
import os
import sys
import time
path='25000'
newpath='results_color'
if not os.path.exists(newpath):
    os.mkdir(newpath)
files=os.listdir(path)

for y in range(len(files)):
    start=time.time()
    os.mkdir(newpath+'/'+files[y])
    newfiles=os.listdir(path+'/'+files[y])
    for x in range(len(newfiles)):

        print(path+'/'+files[y]+'/'+newfiles[x])
        gray1=cv2.imread(path+'/'+files[y]+'/'+newfiles[x])
        fake_gray1=gray1<5
        gray1=255*fake_gray1+(1-fake_gray1)*gray1
        print(path+'/'+files[y]+'/'+newfiles[x])
        gray=gray1[:,:,0]
        r=gray.copy()
        g=gray.copy()
        b=gray.copy()
        mask=(gray<=10)*(gray>=0)
        r=mask*255+(1-mask)*r
        g=mask*245+(1-mask)*g
        b=0*mask+(1-mask)*b

        mask=(gray<=15)*(gray>=10)
        r=mask*255+(1-mask)*r
        g=mask*191+(1-mask)*g
        b=mask*0  +(1-mask)*b

        mask=(gray<=20)*(gray>=15)
        r=mask*255+(1-mask)*r
        g=mask*0+(1-mask)*g
        b=mask*0+(1-mask)*b

        mask=(gray<=25)*(gray>=20)
        r=mask*0+(1-mask)*r
        g=mask*255+(1-mask)*g
        b=mask*0+(1-mask)*b


        mask=(gray<=30)*(gray>=25)
        r=mask*0+(1-mask)*r
        g=mask*205+(1-mask)*g
        b=mask*0+(1-mask)*b

        mask=(gray<=35)*(gray>=30)
        r=mask*0+(1-mask)*r
        g=mask*139+(1-mask)*g
        b=mask*0+(1-mask)*b

        mask=(gray<=40)*(gray>=35)
        r=mask*0+(1-mask)*r
        g=mask*255+(1-mask)*g
        b=mask*255+(1-mask)*b

        mask=(gray<=45)*(gray>=40)
        r=mask*0+(1-mask)*r
        g=mask*215+(1-mask)*g
        b=mask*255+(1-mask)*b

        mask=(gray<=50)*(gray>=45)
        r=mask*63+(1-mask)*r
        g=mask*133+(1-mask)*g
        b=mask*205+(1-mask)*b

        mask=(gray<=55)*(gray>=50)
        r=mask*0+(1-mask)*r
        g=mask*0+(1-mask)*g
        b=mask*255+(1-mask)*b

        mask=(gray<=60)*(gray>=55)
        r=mask*34+(1-mask)*r
        g=mask*34+(1-mask)*g
        b=mask*178+(1-mask)*b

        mask=(gray<=65)*(gray>=60)
        r=mask*0+(1-mask)*r
        g=mask*0+(1-mask)*g
        b=mask*139+(1-mask)*b

        mask=(gray<=70)*(gray>=65)
        r=mask*255+(1-mask)*r
        g=mask*0+(1-mask)*g
        b=mask*255+(1-mask)*b

        mask=(gray<=75)*(gray>=70)
        r=mask*255+(1-mask)*r
        g=mask*48+(1-mask)*g
        b=mask*155+(1-mask)*b

        mask=(gray>75)
        r=mask*250+(1-mask)*r
        g=mask*250+(1-mask)*g
        b=mask*255+(1-mask)*b
        rgb = np.zeros((gray1.shape)).astype("uint8")
        rgb[:,:,0]=r
        rgb[:,:,1]=g
        rgb[:,:,2]=b
        cv2.imwrite(newpath+'/'+files[y]+'/'+newfiles[x],rgb)
    end=time.time()
    print(end-start)
