from itertools import count

import cv2

img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),20)

cv2.imshow("Contours:",img)
cv2.waitKey()
cv2.destroyAllWindows()





