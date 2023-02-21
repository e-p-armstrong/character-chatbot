import pathlib
from PIL import Image
import pytesseract

# Code adapted from https://stackoverflow.com/questions/71624703/convert-bgr-colored-image-to-grayscale-except-one-color?noredirect=1&lq=1
# And from https://stackoverflow.com/questions/65881472/pytesseract-read-coloured-text

import numpy as np
import cv2
import re

# img must be string path to img
def get_text(img): 
    im = Image.open(img)
    width,height = im.size

    left = width / 8
    top = 2.75 * height / 4
    right = 0.90 * width
    bottom = height

    new_im = im.crop((left,top,right,bottom))
    new_im.save('tmp.jpg')
    image = cv2.imread('tmp.jpg')#cv2.imread(img)
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.merge([gray, gray, gray])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 90, 88])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    colored_output = cv2.bitwise_and(image, image, mask=mask) # combine the image with itself (do nothing) but add the mask
    # gray_output = cv2.bitwise_and(gray, gray, mask=255-mask)
    # result = cv2.add(colored_output, gray_output)
    # img = cv2.imread(filename)
    HSV_img = cv2.cvtColor(colored_output,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(HSV_img)
    thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # [1] gets the thresholded image
    cv2.imwrite(f'{img}binary_thresh.jpg', thresh)
    txt = pytesseract.image_to_string(thresh)
    txt = process_text(txt)
    return txt
    # cv2.imwrite('colored_output.jpg',colored_output)
    # return colored_output
    # cv2.waitKey()

# Adapted from https://stackoverflow.com/questions/3939361/remove-specific-characters-from-a-string-in-python
def process_text(txt):
    translation_table = dict.fromkeys(map(ord,'\{\}\\$/%^&^*()[]'),None) # creates a translation dictionary where each character in the list becomes a key with value none
    result = txt.translate(translation_table)
    print(f"\n\nString pre-format: {result}")
    result = re.sub('\n',' ',result) # In the end the main fix was applied by using Chizuru's name as a landmark to kill all the random characters that could appear before it. This involved stripping the newlines BEFORE the main regex format.
    result = re.sub('.+?Chizuru.+?(?=[A-Z])|Chizuru.+?(?=|[A-Z])|(J$)','',result) # Strip the Leading Chizuru and trailing J # TODO strip all chars before leading chizuru
    result = re.sub('\n',' ',result).strip()
    if ((len(result) > 0)):
        if (result[-1] == "J"):
            result = result[:len(result)-1] # hack to kill trailing J, update to use regex
    print(f"\n\nString post-format: {result}")
    return  result# note: replace newlines with spaces


# im = Image.open("test_night.jpg")
# print(im.size)
# width,height = im.size

# left = width / 4
# top = 2.75 * height / 4
# right = width
# bottom = height

# new_im = im.crop((left,top,right,bottom))
# new_im.save('tmp.jpg')

# top = height/5
# left =  width/10
# right = width
# bottom = height#2040


result = get_text('test_night.jpg')
print(result)


images = pathlib.Path("screenshots/").iterdir()

all_text = []
for f in images:
    if "/." not in str(f): # Learn more about the in keyword
        print(f)
        all_text.append(get_text(str(f)))

# print(all_text)

# Next step from here: write a regex and text processing script to deal with output issues like the trailing J, special characters like / and \, and the inconsistent brackets around Chizuru's name.
# Wait a sec, just trim all special characters (brackets and slashes) and find a way to kill the J.

# ideal regex for post-special-character strip: match (remove) the word Chizuru, and everything before and after it that isn't a letter.
# OR match a trailing J.

# Note to self:
# reader = easyocr.Reader(['en'])



# Preprocessing


# What I'm looking for is not optical character recognition, but computer vision, apparently. the difference being that here there's an image with some text on it
# while pytesseract is looking for an image that's just text. How simple. Anyway, look into EasyOCR
# result = reader.readtext('colored_output.jpg', detail=0)
# result2 = reader.readtext('test.jpg', detail=0)
# print(pytesseract.image_to_string(Image.open("test.jpg")))
# print(result2)

# read text from all files
# for f in images:
#     print(pytesseract.image_to_string(Image.open(f)))
#     print(f) # debug
    
