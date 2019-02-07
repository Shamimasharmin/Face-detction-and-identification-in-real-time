# Face-detction-and-identification-in-real-time
Detect and Identify human faces in real time Implement opencv with python to do face detection and face identification: 

Face Detection: Face detection is just pulling out faces in an image or a video. This technique we are using that shows for both face detection and identification. 

Face Identification: Face identification is identify faces in a video or image including the person's name. This takes a little bit extra training, that we have done in here. The training portion and the identification of faces can be absolutely advanced. Advanced means using deep learning libraries such as tensorflow or PI torch. 
Opencv has some built-in features to make those things(advanced) easier. To do this project here used: numpy==1.15.4, opencv==3.4.3.18, pillow=5.3.0, virualenv==16.1.0, python=3.6.7
First here is a base.py file which is just for opening the webcam automatic when I run the code. Now I name it as faces.py:Start with this faces.py file

✔ Step 1: Copy haarcascades from cv2

So, first starting with a new file named as faces.py and in this file first need to write the code of opening the webcam, which have done before and which named as base.py file. Now, following the first step as it said need to get built-in haarcascade classifier. Haarcascade classifier : This is something that is relevant to actually identifying a face in any given frame. Now copy the haarcascade file. So the haarcascade file is where the opencv is located. In opencv there is lib folder and than in lib there is the site-packages than there is cv2. In the cv2 there is data. So, here(in data folder) is the needed cascade files that are required. Copy the data folder than In src folder of the virtualenvironment created a new folder named as cascades and paste it(data folder) in the cascades folder.

✔ Step 2: haarcascade classifier

Now need to get the cascade file named as haarcascade_frontalface_alt2.xml in faces.py file. In the faces.py file declare haarcascade_frontalface as face_cascade and than showing the path where the haarcascade files are. face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

✔ Step 3: Using the face classifier

Use this face cascade to detect the faces in the frame. Before doing that need to convert the frame in gray because cascade works in this way(gray). So, put it in gray scale. gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) To detect the face in the frame: faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3) scalefactor, minneighbors number can be changed if don't get the result as hoped. if scalefactor > 1.5, it is a potentially problem. minNeighbors defines how many objects are detected near the current one before it declares the face found. Next iterate through the faces for (x, y, w, h) in faces: print(x,y,w,h) Now, running the faces.py file and result shows all the values of the faces region of interest in gray = location of the frame roi_gray = gray[y:y+h, x:x+w] To save the image by real time camera: img_item = "8.png" cv2.imwrite(img_item, roi_gray) Now, running faces.py file can see the values of faces like before and also saves only the portion of the face in gray colour and everything removes out. But in the real time camera, it shows coloured frame and saved image in gray colour.

Region of interest by gray frame: It has to do with the actual pixel values for that item(saving image item) . So the pixel values are taking y values and y+h; h is the bottom part that means y coordinate starts and y coordinates end, same for x coordinate starts and x coordinate end . Also can say just taking it into height and the width x=x+w that means x = left side of it than add w which is width. x,y are the starting coordinate and x+w, y+h are the ending coordinates. roi_color = frame[y:y+h, x:x+w] #region of interest for color frame =location of the frame

✔ Step 4: Draw a rectangle

To draw the rectangle in the frame first thing to do is declare what color going to use for the rectangle. Here the rectangle colour is blue for that color=255 and 0 , 0 means full rectangle colour is blue. color = (255, 0, 0) Stroke means how thick the line of the rectangle stroke = 2 #thickness value of the rectangle Grab the frame and drawing on the original color frame not the gray frame than starting the coordinates which are x and y. cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) than the width w and the height h to add. (x,y) are the starting coordinate and x+w, y+h are the ending coordinates as told in step 2. end_cord_x = x + w end_cord_y = y + h

Now, running the faces.py file and see a blue rectangle on face portion only. It detect only the frontal face because of haarcascade_frontalface_alt2.xml file. It doesn't show the rectangle when showing one side portion of the face. In windows powershell it is showing the values of face.

✔ Step 5: Recognizer

Now, start with opening a new file named as faces-train.py: Opencv has the ability to train a recognizer, this is where we could use a deep learned model to predict things as well. There are: tensorflow, scikit-learn, pi-torch, keras. Opencv use the haarcascade classifier to say there is a face right there. Now, need to identify the face, like who the person is. It doesn't work perfectly.But it actually works and not complicated. For this, need to train the images with their identification. So, create directory(folder) named them as the each persons name which is their identification and than save the images for each specific persons in virtualenvironment’s src folder.

✔ Step 6: os.walk for images

To open the image directory here have to create a path of saved images. The images that I want to train for identification. import os Open the directory(folder) of the images where the images are saved. Actually Looking through the directory for png or jpeg files and adding them to some sort of list. BASE_DIR = os.path.dirname(os.path.abspath(file)) image_dir = os.path.join(BASE_DIR, "images") Next to see the images in there for root, dirs, files in os.walk(image_dir): Inside of this for loop do iterate through the files in there for file in files: file type ‘png’ or ‘jpg’ if file.endswith("png") or file.endswith("jpg"): print the path of the files path = os.path.join(root, file) print path Now, running faces-train.py file and got the result in powershell. The directory(folder) named as the persons name with their images and it is showing the path of all the images.

✔ Step 7: Labels from directories

Grab the name of the directory(folder) that it's in to give it a label. The images name doesn’t matter here. The directory of the person’s name folder matters the most and that is going to be a label. Label means each persons containing images folder names. label = os.path.basename(root).replace(" ", "-").lower() print(label, path) Now, running faces-train.py and the result is in powershell. Depending on the directories(folder) the result is showing label and the path of the stored images. Label means the name of the directories. replace(" ", "-").lower(), this is just safeguard which means replacing any sort of space with a dash. This is mainly for if accidentally name the directory incorrectly and than also want to lower case everything on that label. It's not the best safeguard. y_labels = [] #empty list x_train = [] #empty list #y_labels.append(label) # some number #x_train.append(path) # verify this image, turn into a numpy array, conver into GRAY

✔ Step 8: Training image to numpy array

There is a pillow library in python which is the python image library. So, install pillow library in powershell to grab pill.

from PIL import Image this will give image of the path pill_image = Image.open(path).convert("L") #turn into grayscale convert into numpy array so grab numpy import numpy as np image_array = np.array(pill_image, "uint8")#turn into numpy array uint8 means Unsigned integer (0 to 255) Now, running faces-train.py file and the result is in windows powershell. This also showing the labels and path of the saved images. Here, the images are converting into numpy array. This is taking every pixels value and turning it into some sort of numpy array. It shows what actually is in the images, it shows the numbers in the actual image . This is what to train on. Every images are different sizes and every image has its pixel values.

✔ Step 9: Region of interest in training data

import cv2 and grab haarcascade frontalface face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') Doing same face detection inside of the actual image because here(faces-train.py) still need to detect the image. faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3) for (x,y,w,h) in faces: roi = image_array[y:y+h, x:x+w] #region of interest in image array x_train.append(roi) #training data

✔ Step 10: Creating training labels

Know the labels but don't have the number of value associated to it current_id = 0 Every new id that is created, would add one. In another word we need to create a dictionary label_ids = {} #put it into empty dictionary create id like 0,1,2,3 for labels . label means each persons containing images folder named. Those names(labels) are given ids 0,1,2,3... if not label in label_ids: label_ids[label] = current_id current_id += 1 id_ = label_ids[label] print(label_ids) #declare the labels specific id y_labels.append(id_) Now, running faces-train.py file and the result is in the powershell. It is declaring each labels a specific id like 0,1,2,3. Label means the persons folder that are named.

✔ Step 11: Using pickle to save label ids

Need to save the labels, so can use them in faces.py file import pickle with open("labels.pickle", 'wb') as f: #writing bytes pickle.dump(label_ids, f) #f means file, now label ids are saved.

✔ Step 12: Train the opencv recognizer

Now need to train the item itself which haven't built in yet. So, we make our facial recognizer.


LBPH(Local Binary Patterns Histograms) : In LBPH each images is analyzed independently, while the eigenfaces method looks at the dataset as a whole. The LBPH method is somewhat simpler, in the sense that we characterize each image in the dataset locally and when a new unknown image is provided, we perform the same analysis on it and compare the result to each of the images in the dataset. The way which we analyze the images is by characterizing the local patterns in each location in the image. Using LBPH method for face recognition, it will probably work better in different environments and light conditions. However, it will depend on our training and testing data sets. There will need around 10 different images of the person's face in order to be able to recognize him/her. 
recognizer = cv2.face.LBPHFaceRecognizer_create() recognizer.train(x_train, np.array(y_labels)) #converting y labels into numpy array recognizer.save("trainner.yml") Now running the faces-train.py file for training a trainner.yml file will be saved in the same directory src, where faces.py and faces-train.py files have been saved.

Now, Start with opening same previous faces.py file:

✔ Step 13: Implement recognizer

First import the lbph recognizer in this faces.py file recognizer = cv2.face.LBPHFaceRecognizer_create() Now read the training data which is trainner.yml recognizer.read("trainner.yml") Now predict things, the predictions are id_ and confidence and we predict region of interest gray id_, conf = recognizer.predict(roi_gray) conf means : its output from the recognizer, which give the value of confidence level of the recognized face. If the value is 0 its 100% confident but the larger the number the lesser the confidence for that prediction. Now use region of interest with the training if conf>=45 and conf <=85: print(id_), Now run faces.py file. It shows the id no of the face by recognizing it, which we have declared previously the id no for each labels.

✔ Step 14: Load label names from pickle

First import pickle in this faces.py file import pickle It give us actual label by writing these command lines labels = {"person name": 1} with open("labels.pickle", 'rb') as f: og_labels = pickle.load(f) #old label now inverting actual labels labels = {v:k for k,v in og_labels.items()} #this is now new label print(labels[id_]) #it shows the label, label means the name Now, running faces.py file. It detecting a face by using a blue rectangle and recognizing the person in front of the webcamera and showing the name which is declared as label. It also showing the id which has been declared before.

✔ Step 15: Opencv put text

now use a font, name, color(white), stroke font = cv2.FONT_HERSHEY_SIMPLEX name = labels[id_] color = (255, 255, 255) stroke = 2 cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA) Now running faces.py file and got the desire result as it is identifing human face in real time camera.

Again start with same faces-train.py file

✔ Step 16: Resize images for training

Need to resize the images first and than convert them if needed and if didn’t get the desire result. Resize all images in a specific size. size = (550, 550) final_image = pill_image.resize(size, Image.ANTIALIAS)

Again start with same faces.py file

✔ Step 17: Eyes

Detect the eyes by importing haarcascade_eye.xml file eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml') eyes = eye_cascade.detectMultiScale(roi_gray) for (ex,ey,ew,eh) in eyes: cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#draw a rectangle around the eye areas Now, running faces.py file and can see the result in figure 4.15. It is detecting eyes with face identification in real time camera. But if the eyes are closed , it will not be able to detect the eyes. It can also detect multiple human faces in real time. If want to detect smile then same as importing haarcascade_smile.xml file smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
