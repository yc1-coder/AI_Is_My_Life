#图片的水平垂直翻转

import cv2
img = cv2.imread("test.jpg")

flip_h = cv2.flip(img,1)

flip_v = cv2.flip(img,0)

flip_both = cv2.flip(img,-1)

cv2.imshow("Original",img)
cv2.imshow("Flip Horizontal",flip_h)
cv2.imshow("Flip Vertical",flip_v)
cv2.imshow("Flip Both",flip_both)

cv2.waitKey(0)
cv2.destroyAllWindows()
