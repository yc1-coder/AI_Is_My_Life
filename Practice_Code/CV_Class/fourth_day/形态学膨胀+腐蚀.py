import cv2

import numpy as np

img = cv2.imread("test.jpg")

#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#二值化
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#卷积核
kernel = np.ones((5,5),np.uint8)

#腐蚀
erosion = cv2.erode(gray,kernel,iterations =1)

#膨胀
dilation = cv2.dilate(gray,kernel,iterations =1)

cv2.imshow("Binary",binary)
cv2.imshow("Erosion",erosion)
cv2.imshow("Dilation",dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()



