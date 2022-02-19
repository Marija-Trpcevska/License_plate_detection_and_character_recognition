import cv2
import numpy as np
import argparse
import pytesseract
import imutils
import glob
from skimage.segmentation import clear_border

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="python License_plate_recognition.py -i|--image [image]")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V, = cv2.split(hsv)

filtered = cv2.bilateralFilter(V, 13, 17, 17)

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrasted = CLAHE.apply(filtered)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
opened = cv2.morphologyEx(contrasted, cv2.MORPH_OPEN, kernel)

subtracted = cv2.subtract(contrasted, opened)
ret, thresholded = cv2.threshold(subtracted, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

sobel_x_direction = cv2.Sobel(thresholded, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
sobel_y_direction = cv2.Sobel(thresholded, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
abs_sobel_x = cv2.convertScaleAbs(sobel_x_direction)
abs_sobel_y = cv2.convertScaleAbs(sobel_y_direction)
sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

contours = cv2.findContours(sobel.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

database = []
for filename in glob.iglob("Standard_plates" + '**/*.*', recursive=True):
    d = cv2.imread(filename, 0)
    database.append(d)

sift = cv2.SIFT_create()

data = dict()
for i in range(len(database)):
    data[i] = sift.detectAndCompute(database[i], None)

query = []
key = dict()
i = 0
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / float(height)
    if 3 <= aspect_ratio <= 5:
        plate = V[y:y + height, x:x + width]
        query.append(plate)
        key[i] = sift.detectAndCompute(plate, None)
        i += 1

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

licensePlate = None
max_average_matching = -1

for i in range(len(query)):
    average_matching = 0
    for j in range(len(database)):
        if data[j][1] is not None and key[i][1] is not None and len(data[j][0]) >= 2 and len(key[i][0]) >= 2:
            matches = flann.knnMatch(data[j][1], key[i][1], k=2)
            best_ones = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    best_ones.append(m)
            if len(best_ones) >= 4:
                key_pt_query = key[i][0]
                key_pt_data = data[j][0]
                src_pts = np.float32([key_pt_data[i.queryIdx].pt for i in best_ones]).reshape(-1, 1, 2)
                dst_pts = np.float32([key_pt_query[i.trainIdx].pt for i in best_ones]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = [best_ones[i] for i in range(len(best_ones)) if mask[i]]
                average_matching += len(inliers)
    average_matching = average_matching / len(database)
    if average_matching > max_average_matching:
        licensePlate = query[i]
        max_average_matching = average_matching

ROI = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
ROI = clear_border(ROI)
ROI = cv2.resize(ROI, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
ROI = cv2.erode(ROI, None, iterations=1)
ROI = cv2.dilate(ROI, None, iterations=1)
ROI = cv2.threshold(cv2.medianBlur(ROI, 3), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
_, ROI = cv2.threshold(ROI, 0, 255, 1)

cv2.imshow("License Plate", licensePlate)
cv2.waitKey(0)
cv2.imshow("ROI", ROI)
cv2.waitKey(0)
cv2.destroyAllWindows()

license_plate_text = None
if licensePlate is not None:
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(7)
    license_plate_text = pytesseract.image_to_string(ROI, config=options)
    license_plate_text = "".join([c if 48 <= ord(c) <= 57 or 65 <= ord(c) <= 90 else "" for c in license_plate_text]).strip()
    print(license_plate_text)
    text_file = open("Results.txt", "a")
    no_chars = text_file.write(license_plate_text + "\n")
    text_file.close()
