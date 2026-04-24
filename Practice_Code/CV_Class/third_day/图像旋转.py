import cv2
img = cv2.imread("test.jpg")

h,w = img.shape[0:2]

center = (h//2,w//2)

M = cv2.getRotationMatrix2D(center,45,1.0)

rotated = cv2.warpAffine(img,M,(w,h))

cv2.imshow("Original",img)
cv2.imshow("Rotated",rotated)

cv2.waitKey()
cv2.destroyAllWindows()