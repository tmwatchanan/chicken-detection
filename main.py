import numpy as np
import cv2
import re  # Regular Expression
import os
import glob
import imutils
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

TRAINING_SET_DIRECTORY = "Contest\\"
PROCESSED_DIRECTORY = "C:\\Users\\Watchanan\\Desktop\\Processed\\"
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"

RESIZE_IMAGE_WIDTH = 400.
CURRENT_FILENAME = ''
SAVE_IMAGE_EXTENSION = '.jpg'
MIN_MATCH_COUNT = 10
POSITIVE_DIRECTORY = "Positives\\"
NEGATIVES_DIRECTORY = "Negatives\\"
CHICKEN_SIZE = (50, 150)
(winW, winH) = CHICKEN_SIZE
hog = cv2.HOGDescriptor(CHICKEN_SIZE, (20, 60), (10, 30), (10, 30), 9)

def ResizeByWidth(image):
    height, width, depth = image.shape
    imgScale = RESIZE_IMAGE_WIDTH / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    return cv2.resize(image, (int(newX), int(newY)))


def SaveImage(image, string):
    cv2.imwrite(PROCESSED_DIRECTORY + CURRENT_FILENAME + '-' + string + SAVE_IMAGE_EXTENSION, image)


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Read filename list
with open(LIST_FILENAME) as f:
    training_set_list = f.readlines()
training_set_list = [x.strip() for x in training_set_list if x.strip()]

# --------------------------------------------------------

numChickens = len(glob.glob1(POSITIVE_DIRECTORY,"*.bmp"))
# numChickens = 3
numNegatives = len(glob.glob1(NEGATIVES_DIRECTORY,"*.bmp"))
label_train = np.zeros((numChickens + numNegatives, 1))
count = 0
# train positives
for i in range(1, numChickens + 1):
    im = cv2.imread(POSITIVE_DIRECTORY + "chicken_train" + str(i) + ".bmp", 0)
    im = cv2.resize(im, CHICKEN_SIZE)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    cv2.imshow("positives " + str(i), im)
    h = hog.compute(im)
    if count == 0:
        features_train = h.reshape(1, -1)
    else:
        features_train = np.concatenate((features_train, h.reshape(1, -1)), axis=0)

    label_train[count] = 1
    count = count + 1
# train negatives
for filename in os.listdir(NEGATIVES_DIRECTORY):
    if filename.endswith(".bmp"):
        # print(os.path.join(NEGATIVES_DIRECTORY, filename))
        currentFileName = NEGATIVES_DIRECTORY + filename
        im = cv2.imread(currentFileName, 0)
        im = cv2.resize(im, CHICKEN_SIZE)
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)
        features_train = np.concatenate((features_train, h.reshape(1, -1)), axis=0)
        label_train[count] = 0
        count = count + 1

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)  # Linear kernel
# svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function => too complex => overfit!
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE, label_train.astype(np.int32))
#
charlist = "NC"



for IMAGE_FILENAME in training_set_list:
    found = False
    fileName = re.split('\.', IMAGE_FILENAME)
    CURRENT_FILENAME = fileName[0]
    print("> " + CURRENT_FILENAME)

    im_original = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
    # cv2.imshow('original', im)

    im = ResizeByWidth(im_original)



    # cv2.imshow("resized " + IMAGE_FILENAME, im)    # blur = cv2.GaussianBlur(im, (15, 15), 2)
    blur = cv2.GaussianBlur(im, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im)
    # s.fill(155)
    # v.fill(255)
    hsv_image = cv2.merge([h, s, v])
    LOWER_YELLOW = np.array([16, 50, 100], dtype=np.uint8)
    UPPER_YELLOW = np.array([95, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
    opening_se = np.ones((5, 1), np.uint8)
    yellow_mask_opening = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, opening_se)
    opening_se = np.ones((1, 5), np.uint8)
    yellow_mask_opening = cv2.morphologyEx(yellow_mask_opening, cv2.MORPH_OPEN, opening_se)
    erode_se = np.ones((1, 1), np.uint8)  # structuring element
    yellow_mask_erode = cv2.erode(yellow_mask_opening, erode_se, iterations=2)
    yellow_mask_median = cv2.medianBlur(yellow_mask_erode, 5)
    # cv2.imshow('yellow mask (binary)', yellow_mask)
    yellow_res = cv2.bitwise_and(im, im, mask=yellow_mask_median)
    # red_res = cv2.bitwise_and(im, im, mask=red_mask)
    # cv2.imshow('yellow mask (color)', yellow_res)

    yellow_res_bboxes = yellow_res.copy()

    # temp, contours, hierarchy = cv2.findContours(yellow_mask_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # # cv2.drawContours(yellow_res, contours, -1, (0, 0, 255), 2)
    # boundingBoxes = []
    # for cnt in contours:
    #     bRect = cv2.boundingRect(cnt)
    #     boundingBoxes.append(bRect)
    # print("length of boundingBoxes = ", len(boundingBoxes))
    # boundingBoxes = np.array(boundingBoxes)
    # pick = non_max_suppression_slow(boundingBoxes, 0.4)
    # print("length of pick = ", len(pick))
    # cropped_roi = []
    # cnt_count = 1
    # for cnt in pick:
    #     x, y, w, h = cnt
    #     cv2.rectangle(yellow_res_bboxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.putText(yellow_res_bboxes, str(cnt_count), (x-30, y), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
    #     cnt_count = cnt_count + 1
    #     crop_res = im[y:y + h, x:x + w]
    #     cropped_roi.append(crop_res)
    #     # im_gray = cv2.resize(crop_res, (50, 50))
    #     # im_gray = (im_gray / 16).astype(np.uint8)
    # cv2.imshow(CURRENT_FILENAME + " yellow_res_bboxes", yellow_res_bboxes)
    #
    # print("length of cropped_roi = ", len(cropped_roi))
    # roi_count = 1
    # for roi in cropped_roi:
    #     im = roi
    #     # cv2.imshow("roi #" + str(roi_count), im)
    #     roi_count = roi_count + 1
    #
    #     im = cv2.resize(im, CHICKEN_SIZE)
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     im = cv2.GaussianBlur(im, (3, 3), 0)
    #     h = hog.compute(im)
    #     # _,result,_,_ = knn.findNearest(h.reshape(1,-1).astype(np.float32),1)
    #     result = svm.predict(h.reshape(1, -1).astype(np.float32))[1]
    #     cv2.imshow("roi#" + str(roi_count) + charlist[result[0][0].astype(int)], im)
    #     # cv2.moveWindow(charlist[result[0][0].astype(int)],100+((im_id-1)%5)*70,np.floor((im_id-1)/5).astype(int)*150)

   # loop over the image pyramid
    roi_count = 1
    for resized in pyramid(im, scale=1.2):
        print("scale changed")
        # loop over the sliding window for each layer of the pyramid
        if found:
            break
        for (x, y, window) in sliding_window(resized, stepSize=15, windowSize=(winW, winH)):
            print("x=" + str(x) + ", y=" + str(y) + " window")
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            im_window = clone[y:(y + winH), x:(x + winW)]
            # cv2.imshow(str(x) + "," + str(y) + " window", im_window)
            # cv2.waitKey(1)
            im_window = cv2.cvtColor(im_window, cv2.COLOR_RGB2GRAY)
            im_window = cv2.GaussianBlur(im_window, (3, 3), 0)
            h = hog.compute(im_window)
            result = svm.predict(h.reshape(1, -1).astype(np.float32))[1]
            if charlist[result[0][0].astype(int)] == "C":
                cv2.imshow(CURRENT_FILENAME + "-> roi#" + str(roi_count) + charlist[result[0][0].astype(int)], im_window)
                # found = True
                # break
            roi_count = roi_count + 1
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)

    # SaveImage(yellow_res_bboxes, 'yellow_mask')


    # GLCM
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im_gray = cv2.resize(im_gray, (50, 50))
    # im_gray = (im_gray / 16).astype(np.uint8)


    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         break

# test
# for im_id in range(1, numChickens + 1):
#     im = cv2.imread(POSITIVE_DIRECTORY + "chicken_train" + str(im_id) + ".bmp", 0)
#
#     im = cv2.resize(im, CHICKEN_SIZE)
#     im = cv2.GaussianBlur(im, (3, 3), 0)
#     h = hog.compute(im)
#     # _,result,_,_ = knn.findNearest(h.reshape(1,-1).astype(np.float32),1)
#     result = svm.predict(h.reshape(1, -1).astype(np.float32))[1]
#     cv2.imshow(str(im_id)+"="+charlist[result[0][0].astype(int)],im)
#     cv2.moveWindow(str(im_id)+"="+charlist[result[0][0].astype(int)],100+((im_id-1)%5)*70,np.floor((im_id-1)/5).astype(int)*150)

# --------------------------------------------------------

print("DONE")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
