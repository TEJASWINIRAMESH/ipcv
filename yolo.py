import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
    exit()

smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

magnitude = cv2.magnitude(sobel_x, sobel_y)
magnitude = cv2.convertScaleAbs(magnitude)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X Gradient')

plt.subplot(1, 3, 2)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y Gradient')

plt.subplot(1, 3, 3)
plt.imshow(magnitude, cmap='gray')
plt.title('Edge Magnitude (Combined)')

plt.show()

cv2.imwrite('sobel_edge_magnitude.jpg', magnitude)

cv2.imshow("Edge Magnitude", magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
