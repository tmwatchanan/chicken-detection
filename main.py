import numpy as np
import cv2
import re # Regular Expression

TRAINING_SET_DIRECTORY = 'Contest\\'
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"

RESIZE_IMAGE_WIDTH = 400.

IMAGE_FILENAME = '1.jpg'

def ResizeByWidth(image):
    height, width, depth = image.shape
    imgScale = RESIZE_IMAGE_WIDTH / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    return cv2.resize(image, (int(newX), int(newY)))


# with open(LIST_FILENAME) as f:
#     training_set_list = f.readlines()
# training_set_list = [x.strip() for x in training_set_list if x.strip()]

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

# 01.jpg
im1 = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
# cv2.imshow("original " + IMAGE_FILENAME, im1)

resized_image = ResizeByWidth(im1)
cv2.imshow("resized " + IMAGE_FILENAME, resized_image)

# --------------------------------------------------------

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp1, des1 = surf.detectAndCompute(resized_image, None)

img1 = cv2.drawKeypoints(resized_image,kp1,None,(255,0,0),4)
cv2.imshow("Keypoints " + IMAGE_FILENAME, img1)

# ---------------------------------------------------------

IMAGE_FILENAME = "chicken.jpg"
im2 = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
resized_image = ResizeByWidth(im2)
# cv2.imshow("resized " + IMAGE_FILENAME, resized_image)

# Find keypoints and descriptors directly
kp2, des2 = surf.detectAndCompute(resized_image, None)

img2 = cv2.drawKeypoints(resized_image,kp2,None,(255,0,0),4)
cv2.imshow("Keypoints " + IMAGE_FILENAME, img2)

# ---------------------------------------------------------

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1,des2)
# # matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
#
# cv2.imshow("Keypoints " + IMAGE_FILENAME, img3)

# --------------------------------------------------------

# im19 = cv2.imread(TRAINING_SET_DIRECTORY + training_set[5])
# cv2.imshow("test image", im19)

# for fileName in training_set:
#     im = cv2.imread(TRAINING_SET_DIRECTORY + fileName)
#     cv2.imshow(fileName, im)
#     # print(fileName)

# for i in range(1, ):
#     print('[READ]: ' + IMAGE_RELATIVE_PATH + IMAGE_FILENAME + str(i) + '.jpg')
#     coin.append(cv2.imread(IMAGE_RELATIVE_PATH + IMAGE_FILENAME + str(i) + '.jpg'))
#     coin_hsv.append(cv2.cvtColor(coin[i - 1], cv2.COLOR_BGR2HSV))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
