import interface_feu
import cv2

img = cv2.imread("b.jpg")
(x1, y1, x2, y2), _ = interface_feu.first_fire_pass(img)
print(interface_feu.confirm_fire(img, x1, y1, x2, y2))

