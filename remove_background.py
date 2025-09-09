import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('image4.jpg', 1)

if img is None:
    print("Изображение не загружено")
    exit()

# преобразуем изображение из BGR в HSV 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# определяем диапазон цвета фона
lower_color = np.array([35, 40, 40]) # нижняя граница зеленого
upper_color = np.array([85, 255, 255]) # верхняя граница зеленого

# создаем маску и инвертируем ее
mask = cv2.inRange(img_hsv, lower_color, upper_color)
mask = cv2.bitwise_not(mask)

# применяем маску к изображению и выводим результат
result = cv2.bitwise_and(img, img, mask=mask)
result = result[:,:,::-1]

plt.imshow(result)
plt.show()
