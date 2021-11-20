import os 
import cv2 as cv 
import keras 
import numpy as np 
from keras.models import load_model 

labels = ["0", "1", "2", "3", "4"] + ["5", "6", "7", "8", "9"]

if __name__ == "__main__":
    m = load_model("./models/model_06_0.9857.h5")
    m.summary()

    capture = cv.VideoCapture(0)
    win_w = 400; win_h = 400;
    off_x = ( 640 - win_w ) // 2
    off_y = ( 480 - win_h ) // 2
    while(1):
        frame, raw = capture.read()
        roi = raw[off_y:off_y+win_h, off_x:off_x+win_w]
        img = cv.resize(roi, (32,32))[np.newaxis, ..., (2,1,0)] / 128.0 - 1
        #cv.CvtColor(img, cv.COLOR_BGR2RGB)
        result = m.predict(img)[0]
        idx = np.argsort(result)[::-1]
        print("==============Top 5 is============== :")
        for i in idx[:5]:
            print("%s : %.4f"%(labels[i], result[i]))
        cv.rectangle(raw, (off_x - 1, off_y - 1), (off_x + win_w + 1, off_y + win_h + 1), color=(0, 255, 0))
        cv.imshow("img", raw)
        #cv.imshow("roi", roi)
        if cv.waitKey(1000) == ord('q'):
            break