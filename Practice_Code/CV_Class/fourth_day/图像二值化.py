import cv2

img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

cv2.imshow("Original",img)
cv2.imshow("Gray",gray)
cv2.imshow("Binary",binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
