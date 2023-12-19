import cv2
import os
import numpy as np
from PIL import Image
import pickle

#safe the path of the current dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "image")

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# recognizer = cv2.face_LBPHFaceRecognizer.create()

current_id = 0
label_ids = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith('jpg') or file.endswith('png'):
			path = os.path.join(root, file)
			#os.path.dirname(path) is the same as root
			label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()
			# print(label,path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			# print(label_ids)
			# y_label.append(label)    # we want to get some number
			# x_train.append(path)     # we want to verify this image and then turn them into gray image
			pil_image = Image.open(path).convert("L")   #grayscale
			image_array = np.array(pil_image, "uint8")  #convert to a numpy array
			# print(image_array)
			faces=facedetect.detectMultiScale(image_array,1.5,5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)

# print(y_label)
# print(x_train)

with open('labels.pickle', 'wb') as f:
	pickle.dump(label_ids, f)

# recognizer.train(x.train, np.array(y_label))
# recognizer.save("trainner.yml")