import os 
import cv2 as cv 
import numpy as np 
import random
make = True 
check = True

def add(img,y,x):
    img = cv.resize(img, (x, y))
    img0  = np.ones([32,32,3],dtype=np.uint8)*255

    i = random.randint(0,32-y)
    j = random.randint(0,32-x)
    img0[i:i+y,j:j+x,:]  = img
    return img0
def contrast_brighten_demo(image,c,b):#c为对比度 b 为亮度
    h,w,ch=image.shape #获取图像尺寸
    blank =np.zeros([h,w,ch],image.dtype) #创建一个和图像尺寸相同的纯黑图片
    #或者blank=np.zeros_like(image,image.dtype)
    dst=cv.addWeighted(image,c,blank,1-c,b)#调整权重
    return dst

def change(img):
    global num
    r = random.randint(8, 32)
    img = cv.resize(img, (r, r))
    img = cv.resize(img, (32, 32))
    for i in range(1,6):
        r1 = random.uniform(0.8,2)
        r2 = random.randint(1,100)
        img1= contrast_brighten_demo(img, r1, r2)
        x = random.randint(16,32)
        y = random.randint(16,32)
        img1 = add(img1, y, x)
        noisy = np.random.randint(0,i*3,(img1.shape),dtype=np.uint8)
        img1 =img1 - noisy
        num += 1
        # cv2.imwrite("%d.jpg" % num, img1)

if __name__ == "__main__":

    if make:
        all_data = []
        all_label = []
        numpy_data = []
        numpy_label = []
        numpy_flag = 0
        for i in range(8):
            num = 0
            path = './picture1/%d'%(i+1)
            for f in os.listdir(path):
                extension = os.path.splitext(f)[-1]
                if ( extension == '.jpg'):
                    img = cv.imread(os.path.join(path, f))
                    try:
                        img = cv.resize(img, (32,32))[...,(2,1,0)] # opencv read as bgr, but we need rgb
                        all_data.append(img)
                        all_label.append((i))
                        r = random.randint(8, 32)
                        img = cv.resize(img, (r, r))
                        img = cv.resize(img, (32, 32))
                        cv.imwrite("dataset/{}/{}.jpg".format(i+1,num),img)
                        num+=1
                        for a in range(1, 6):
                            r1 = random.uniform(0.8, 2)
                            r2 = random.randint(1, 100)
                            img1 = contrast_brighten_demo(img, r1, r2)
                            x = random.randint(16, 32)
                            y = random.randint(16, 32)
                            img1 = add(img1, y, x)
                            noisy = np.random.randint(0, a * 3, (img1.shape), dtype=np.uint8)
                            img1 = img1 - noisy
                            all_data.append(img1)
                            all_label.append((i))
                            cv.imwrite("dataset/{}/{}.jpg".format(i+1,num),img1)
                            num += 1
                    except:
                        continue
                if (extension == '.npy'):
                    npy_file = os.path.join(path, f)
                    tmp = np.load(npy_file)
                    numpy_data.append(tmp)
                    numpy_label += [(i)] * (len(tmp))
                    numpy_flag = 1
        if 1==numpy_flag:          
            npy_tmp = numpy_data[0]
            for npy in numpy_data[1:]:
                npy_tmp = np.vstack([npy_tmp, npy])

            all_data = np.asarray(all_data)
            all_data = np.vstack([all_data, npy_tmp]) if len(all_data) else npy_tmp

            all_label = np.asarray(all_label + numpy_label)
            all_label = np.asarray(all_label)
        else:
            all_data = np.asarray(all_data)
            all_label = np.asarray(all_label)
            
        np.save("x", all_data)
        np.save("y", all_label)
    if check:
        x = np.load("x.npy")
        y = np.load("y.npy")
        label = ["1", "2", "3", "4"] + ["5", "6", "7", "8"]
        for d,idx in zip(x, y):
            print("Class %s"%label[idx])
            d = cv.resize(d, (32,32))[...,(2,1,0)]
            cv.imshow("img", d)
            # cv.waitKey(1)
