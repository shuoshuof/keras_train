import random

import cv2
import numpy as np
num = 0

def add(img,y,x):
	img = cv2.resize(img, (x, y))
	img0  = np.ones([32,32,3],dtype=np.uint8)*255

	i = random.randint(0,32-y)
	j = random.randint(0,32-x)
	img0[i:i+y,j:j+x,:]  = img
	return img0
def contrast_brighten_demo(image,c,b):#c为对比度 b 为亮度
	h,w,ch=image.shape #获取图像尺寸
	blank =np.zeros([h,w,ch],image.dtype) #创建一个和图像尺寸相同的纯黑图片
	dst=cv2.addWeighted(image,c,blank,1-c,b)#调整权重
	return dst

def change(img):
	global num
	r = random.randint(8, 32)
	img = cv2.resize(img, (r, r))
	img = cv2.resize(img, (32, 32))
	for i in range(1,6):
		r1 = random.uniform(0.8,2)
		r2 = random.randint(1,100)
		img1= contrast_brighten_demo(img, r1, r2)
		x = random.randint(16,32)
		y = random.randint(16,32)
		img1 = add(img1, y, x)
		noisy = np.random.randint(0,i*3,(img1.shape),dtype=np.uint8)
		img1 =img1 - noisy
		# cv2.imwrite("%d.jpg" % num, img1)
		num+=1
# img = cv2.imread("picture/1/1.jpg")
#
# change(img)
# img1 = contrast_brighten_demo(img,0.5,100)
# cv2.imshow("img",img)
# cv2.imshow("img1",img1)
# cv2.imwrite("img1.jpg",img1)
# cv2.imwrite("img.jpg",img)
# cv2.waitKey(0)