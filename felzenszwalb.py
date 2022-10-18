##-*- coding:utf-8 -*-

import cv2
#import numpy as np

## 入力画像を読み込み
#img = cv2.imread("LenaRGB.bmp")

## グレースケール変換
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

## 方法3
#gray_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#gray_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#dst = np.sqrt(gray_x ** 2 + gray_y ** 2)

#img_gray = 255 - dst

#dst = cv2.GaussianBlur(img_gray, (9, 9), 0).astype('uint8')
#dst = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,8)


## 結果を出力
#cv2.imwrite("Sobel.png", dst)

#import cv2
#import numpy as np

 
#image_path = "LenaRGB.bmp"
#img = cv2.imread(image_path)
 
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.merge((gray, gray, gray), img)
 
#kernel = np.ones((4,4),np.uint8)
#dilation = cv2.dilate(img,kernel,iterations = 1)
 
#diff = cv2.subtract(dilation, img)
 
#negaposi = 255 - diff 
#cv2.imwrite("negaposi.png", negaposi)
from skimage import segmentation
from PIL import Image
import numpy as np

def label2rgb(label_field, image):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    for label in labels:
        mask = (label_field == label).nonzero()
        std = np.std(image[mask])
        if std < 20:
            color = image[mask].mean(axis=0)
        elif 20 < std < 40:
            mean = image[mask].mean(axis=0)
            median = np.median(image[mask], axis=0)
            color = 0.5*mean + 0.5*median
        elif 40 < std:
            color = np.median(image[mask], axis=0)
        out[mask] = color
    return out


img_path = "/content/CartoonTrans/data/val/photo/0.jpg"
image = np.array(Image.open(img_path))
image = cv2.medianBlur(image,3)
seg_labels = segmentation.felzenszwalb(image, scale=10, sigma=1, min_size=100)
img_cvtcolor = label2rgb(seg_labels, image)
#result = np.concatenate((image,img_cvtcolor), axis=1)
#print(result.shape)
PIL_image = Image.fromarray(img_cvtcolor)
PIL_image.save("felzenszwalb.png")