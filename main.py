import numpy as np
import cv2
import re # Regular Expression

TRAINING_SET_DIRECTORY = 'Contest\\'
IMAGE_FILENAME = ''
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"

# with open(LIST_FILENAME) as f:
#     training_set_list = f.readlines()
# training_set_list = [x.strip() for x in training_set_list if x.strip()]

with open(RESULT_FILENAME) as f:
    training_set_result = f.readlines()
training_set_result = [x.strip() for x in training_set_result if x.strip()]

for str in training_set_result:
    result_list = re.split(':|,',str)
    if result_list[1] != "none":
        x = int(result_list[1])
        y = int(result_list[2])
        w = int(result_list[3])
        h = int(result_list[4])
        im = cv2.imread(TRAINING_SET_DIRECTORY + result_list[0])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow(result_list[0], im)
        print(result_list)
    else:
        print(result_list[0])


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
