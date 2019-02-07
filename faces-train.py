import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # this is to find the path of faces-train.py file
image_dir = os.path.join(BASE_DIR, "images") #looks for the folder of saved images

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()#Local Binary Patterns Histograms 

current_id = 0
label_ids = {} #empty dictionary
y_labels = [] #empty list
x_train = [] #empty list

for root, dirs, files in os.walk(image_dir):
	for file in files: #iterate through the files
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file) #print the path of the files
			label = os.path.basename(root).replace(" ", "-").lower() #label means name of the folder of the saved images of each persons
			print(label, path) #print the folder names of each persons images with path, here folder is label
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			#print(label_ids) 
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a numpy array, GRAY
			pill_image = Image.open(path).convert("L") # grayscale, pill is the python image library
			#size = (550, 550) #to resize images if needed
  			#final_image = pill_image.resize(size, Image.ANTIALIAS)#resize the saved images to get good result
			image_array = np.array(pill_image, "uint8") #convert into numpy array
			print(image_array)#convert the images into numbers , print the numbers of the images
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
