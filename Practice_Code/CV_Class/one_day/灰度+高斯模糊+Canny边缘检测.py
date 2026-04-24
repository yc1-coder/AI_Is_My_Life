# 1-先看懂这段代码（2 分钟）
# 2-关掉参考，凭记忆手写第 1 遍
# 3-对照改错，标记忘记的地方（3 分钟）
# 4-不看任何提示，手写第 2 遍
# 5-总量：1 个案例 × 2 遍完整默写
# 6-时长：20～30 分钟
# 7-过关标准：不看参考能从头到尾流畅写完不出错

import cv2
img = cv2.imread("../third_day/test.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(blur,50,150)

cv2.imshow("Original",img)
cv2.imshow("Gray",gray)
cv2.imshow("Canny",canny)

cv2.waitKey()
cv2.destroyAllWindows()