import cv2
import numpy as np

# 假设有一个轮廓
contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])

# 计算外接矩形
x, y, w, h = cv2.boundingRect(contour)

print(f"左上角坐标: ({x}, {y})")
print(f"宽度: {w}, 高度: {h}")
print(f"右下角坐标: ({x+w}, {y+h})")

# 在图像上绘制矩形
img = cv2.imread("image.jpg")
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
