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


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh=0.4):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# Read filename list
with open(LIST_FILENAME) as f:
    training_set_list = f.readlines()
training_set_list = [x.strip() for x in training_set_list if x.strip()]


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
    opening_se = np.ones((5,1),np.uint8)
    yellow_mask_opening = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, opening_se)
    opening_se = np.ones((1,5),np.uint8)
    yellow_mask_opening = cv2.morphologyEx(yellow_mask_opening, cv2.MORPH_OPEN, opening_se)
    yellow_mask_median = cv2.medianBlur(yellow_mask_opening, 5)
    # cv2.imshow('yellow mask (binary)', yellow_mask)
    yellow_res = cv2.bitwise_and(im, im, mask=yellow_mask_median)
    # cv2.imshow('yellow mask (color)', yellow_res)

    temp, contours, hierarchy = cv2.findContours(yellow_mask_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(yellow_res, contours, -1, (0, 0, 255), 2)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(yellow_res, (x, y), (x + w, y + h), (0, 0, 255), 2)
    SaveImage(yellow_res, 'yellow_mask')

    # GLCM
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im_gray = cv2.resize(im_gray, (50, 50))
    # im_gray = (im_gray / 16).astype(np.uint8)


    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         break

# --------------------------------------------------------

print("DONE")
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         exit()
