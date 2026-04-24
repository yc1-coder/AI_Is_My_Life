import cv2

img = cv2.imread("../third_day/test.jpg")

width,height = img.shape[:2]

new_size = 640,480

img_size = cv2.resize(img,new_size,interpolation=cv2.INTER_LINEAR)

cv2.imshow("img",img)

cv2.imshow("img_size",img_size)

cv2.imwrite("../third_day/resized.jpg", img_size)

cv2.waitKey(0)

cv2.destroyAllWindows()