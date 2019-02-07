import numpy as np 
import cv2
import pickle
# haaarcascade is pre-trained model/xml file , it is provided by opencv
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') #This is relevant to identifying a face in any given frame
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the frame into gray
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
	for (x, y, w, h) in faces:
		#print(x,y,w,h) #onleft side all the values where the face is
		roi_gray = gray[y:y+h, x:x+w] #region of interest for gray frame =location of the frame
		roi_color = frame[y:y+h, x:x+w]  #region of interest for color frame =location of the frame

 
        # recognize? deep learned model predict
		id_, conf = recognizer.predict(roi_gray) # conf means : its output from the recognizer.. which give the value of confidence level of the recognized face.. if the value is 0 its 100% confident but the larger the number the lesser the confidence for that prediction﻿
		if conf>=45 and conf <=85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)#1 is font size


		img_item = "8.png" #to save the image
		cv2.imwrite(img_item, roi_gray)#save the portion of the face and everything removes out

		color = (255, 0, 0) #BGR 0-255, rectangle color blue
		stroke = 2 #thickness value of the rectangle
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)#grab the frame in color
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# Display the resulting frame
	cv2.imshow('frame', frame) #imgshow,it is declare as frame that's why it doesn't shows gray color
	if cv2.waitKey(20) & 0xFF == ord('q'):#to close the frame just press q
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()