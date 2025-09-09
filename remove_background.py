import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image4.jpg', 1)

if img is None:
    print("Изображение не загружено")
    exit()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_color = np.array([35, 40, 40])
upper_color = np.array([85, 255, 255])

mask = cv2.inRange(img_hsv, lower_color, upper_color)
mask = cv2.bitwise_not(mask)

result = cv2.bitwise_and(img, img, mask=mask)
result = result[:,:,::-1]

plt.imshow(result)
plt.show()
