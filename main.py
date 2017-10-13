import numpy as np
import cv2
import re  # Regular Expression
from skimage.feature import greycomatrix, greycoprops

TRAINING_SET_DIRECTORY = "Contest\\"
PROCESSED_DIRECTORY = "C:\\Users\\Watchanan\\Desktop\\Processed\\"
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"

RESIZE_IMAGE_WIDTH = 400.
CURRENT_FILENAME = ''
SAVE_IMAGE_EXTENSION = '.jpg'


def ResizeByWidth(image):
    height, width, depth = image.shape
    imgScale = RESIZE_IMAGE_WIDTH / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    return cv2.resize(image, (int(newX), int(newY)))

def SaveImage(image, string):
    cv2.imwrite(PROCESSED_DIRECTORY + CURRENT_FILENAME + '-' + string + SAVE_IMAGE_EXTENSION, image)


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

for IMAGE_FILENAME in training_set_list:
    fileName = re.split('\.', IMAGE_FILENAME)
    CURRENT_FILENAME = fileName[0]

    im_original = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
    # cv2.imshow('original', im)

    im = ResizeByWidth(im_original)
    # cv2.imshow("resized " + IMAGE_FILENAME, im)

    blur = cv2.GaussianBlur(im, (15, 15), 2)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im)
    # s.fill(255)
    # v.fill(255)
    hsv_image = cv2.merge([h, s, v])
    LOWER_YELLOW = np.array([16, 120, 100], dtype=np.uint8)
    UPPER_YELLOW = np.array([95, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
    kernel = np.ones((5,5),np.uint8)
    yellow_mask_opening = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask_median = cv2.medianBlur(yellow_mask_opening, 5)
    # cv2.imshow('yellow mask (binary)', yellow_mask)
    yellow_res = cv2.bitwise_and(im, im, mask=yellow_mask_median)
    # cv2.imshow('yellow mask (color)', yellow_res)
    SaveImage(yellow_res, 'yellow_mask')

    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         break

# --------------------------------------------------------

print("DONE")
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         exit()
