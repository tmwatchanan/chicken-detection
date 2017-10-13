import cv2
import re  # Regular Expression
import os, errno

TRAINING_SET_DIRECTORY = "Contest\\"
SOLUTIONS_DIRECTORY = "Solutions\\"
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"
SAVE_IMAGE_EXTENSION = '.jpg'

# Read list of training images
with open(RESULT_FILENAME) as f:
    training_set_result = f.readlines()
training_set_result = [x.strip() for x in training_set_result if x.strip()]

try:
    os.makedirs(SOLUTIONS_DIRECTORY)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# See solutions
for str in training_set_result:
    result_list = re.split(':|,',str)
    if result_list[1] != "none":
        x = int(result_list[1])
        y = int(result_list[2])
        w = int(result_list[3])
        h = int(result_list[4])
        im = cv2.imread(TRAINING_SET_DIRECTORY + result_list[0])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(SOLUTIONS_DIRECTORY + 'solution-' + result_list[0] + SAVE_IMAGE_EXTENSION, im)
        # cv2.imshow(result_list[0], im)
        print(result_list)
    else:
        print(result_list[0])
