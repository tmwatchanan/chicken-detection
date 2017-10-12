import numpy as np
import cv2
import re  # Regular Expression

TRAINING_SET_DIRECTORY = 'Contest\\'
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"

RESIZE_IMAGE_WIDTH = 400.


def ResizeByWidth(image):
    height, width, depth = image.shape
    imgScale = RESIZE_IMAGE_WIDTH / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    return cv2.resize(image, (int(newX), int(newY)))


with open(LIST_FILENAME) as f:
    training_set_list = f.readlines()
training_set_list = [x.strip() for x in training_set_list if x.strip()]

# Read list of training images
# with open(RESULT_FILENAME) as f:
#     training_set_result = f.readlines()
# training_set_result = [x.strip() for x in training_set_result if x.strip()]

# See solutions
# for str in training_set_result:
#     result_list = re.split(':|,',str)
#     if result_list[1] != "none":
#         x = int(result_list[1])
#         y = int(result_list[2])
#         w = int(result_list[3])
#         h = int(result_list[4])
#         im = cv2.imread(TRAINING_SET_DIRECTORY + result_list[0])
#         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.imshow(result_list[0], im)
#         print(result_list)
#     else:
#         print(result_list[0])

# --------------------------------------------------------

IMAGE_FILENAME = '7.jpg'

# 01.jpg
im1_original = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
# cv2.imshow("original " + IMAGE_FILENAME, im1_original)

im1 = ResizeByWidth(im1_original)
cv2.imshow("resized " + IMAGE_FILENAME, im1)

blur = cv2.GaussianBlur(im1, (15, 15), 2)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(im1)
# s.fill(110)
# v.fill(255)
hsv_image = cv2.merge([h, s, v])
LOWER_YELLOW = np.array([35, 110, 180], dtype=np.uint8)
UPPER_YELLOW = np.array([90, 255, 255], dtype=np.uint8)
yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
cv2.imshow('yellow mask (binary)', yellow_mask)
yellow_res = cv2.bitwise_and(im1, im1, mask= yellow_mask)
cv2.imshow('yellow mask (color)', yellow_res)

# --------------------------------------------------------

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
