# -*- coding: cp1254 -*-
# USAGE
# python linear_classifier.py --dataset kaggle_dogs_vs_cats

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
        
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])


	if imutils.is_cv2():
		hist = cv2.normalize(hist)


	else:
		cv2.normalize(hist, hist)


	return hist.flatten()




print("[INFO] describing images...")
imagePaths = list(paths.list_images("C:\\Users\\ZİYA\\Desktop\\Datasets\\train1"))


data = []
labels = []


for (i, imagePath) in enumerate(imagePaths):

	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = extract_color_histogram(image)
	data.append(hist)
	labels.append(label)


	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))


le = LabelEncoder()
labels = le.fit_transform(labels)


print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.25, random_state=1000)


print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
joblib.dump(model, "D:/svm1.model")

print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
datas = []
imre = cv2.imread("C:\\Users\\ZİYA\\Desktop\\Datasets\\red1\\1051.png")
imre_hist = extract_color_histogram(imre)
datas.append(imre_hist)
print model.predict(datas)
print(classification_report(testLabels, predictions,
	target_names=le.classes_))
