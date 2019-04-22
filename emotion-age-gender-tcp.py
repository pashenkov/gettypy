import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, \
	ZeroPadding2D, MaxPooling2D, AveragePooling2D, \
	Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir
import socket

#set server variables
TCP_IP = '127.0.0.1'
TCP_PORT = 7001
BUFFER_SIZE = 512

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()

# print connection address when someone connects
print('Connection address:', addr)


enableGenderIcons = True

male_icon = cv2.imread("../dataset/male.jpg")
male_icon = cv2.resize(male_icon, (40, 40))

female_icon = cv2.imread("../dataset/female.jpg")
female_icon = cv2.resize(female_icon, (40, 40))

face_cascade = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def loadEmotionModel():


	# define CNN model
	model = Sequential()

	# 1st convolution layer
	model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	# 2nd convolution layer
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	# fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	num_classes = 7
	model.add(Dense(num_classes, activation='softmax'))
	return model

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	return model

def ageModel():
	model = loadVggFaceModel()
	
	base_model_output = Sequential()
	base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)
	
	age_model = Model(inputs=model.input, outputs=base_model_output)
	age_model.load_weights("../models/age_model_weights.h5")
	return age_model

def genderModel():
	model = loadVggFaceModel()
	
	base_model_output = Sequential()
	base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	gender_model = Model(inputs=model.input, outputs=base_model_output)
	gender_model.load_weights("../models/gender_model_weights.h5")
	return gender_model

def emotionModel():

	#emotion_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
	emotion_model = loadEmotionModel()
	emotion_model.load_weights('../models/facial_expression_model_weights.h5') #load weights
	return emotion_model

age_model = ageModel()
gender_model = genderModel()
emotion_model = emotionModel()

# age model has 101 outputs and its outputs will be multiplied by its index label.
# sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

cap = cv2.VideoCapture(0) #capture webcam

primary_emotion = "neutral"

while(True):
	ret, img = cap.read()
	#img = cv2.resize(img, (640, 360))

	canvas = np.zeros((640, 480, 3), dtype="uint8")

	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x, y, w, h) in faces:
		if w > 50:  # 130: #ignore small faces
			
			#extract detected face
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			#cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

			# detect emotions
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255 # normalize all pixels to scale of [0, 1]

			predictions = emotion_model.predict(img_pixels)

			# find max of array
			max_index = np.argmax(predictions[0])

			# using max_index to assign primary_emotion
			# TO DO: come up with a better way to determine primary_emotion
			# work with data to average predictions over time, etc.
			primary_emotion = emotions[max_index]

			cv2.putText(img, primary_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

			# '''
			# draw emotion probabilities
			for i in range(len(emotions)):
				# construct the label text
				text = "{}: {:.2f}%".format(emotions[i], predictions[0][i] * 100)
				bar_width = int(predictions[0][i] * 300)
				cv2.rectangle(canvas, (7, (i * 35) + 5),
					(bar_width, (i * 35) + 35), (0, 0, 255), -1)
				cv2.putText(canvas, text, (10, (i * 35) + 23),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45,
					(255, 255, 255), 2)
			# '''

			try:
				#age gender data set has 40% margin around the face. expand detected face.
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			try:
				'''
				#vgg-face expects inputs (224, 224, 3)
				detected_face = cv2.resize(detected_face, (224, 224))
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				#find out age and gender
				age_distributions = age_model.predict(img_pixels)
				apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
				
				gender_distribution = gender_model.predict(img_pixels)[0]
				gender_index = np.argmax(gender_distribution)
				
				if gender_index == 0: gender = "F"
				else: gender = "M"
			
				#background for age gender declaration
				info_box_color = (46,200,255)
				#triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
				
				#labels for age and gender
				cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
				if enableGenderIcons:
					if gender == 'M': gender_icon = male_icon
					else: gender_icon = female_icon
					
					img[y-75:y-75+male_icon.shape[0], x+int(w/2)-45:x+int(w/2)-45+male_icon.shape[1]] = gender_icon
				else:
					cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				'''

			except Exception as e:
				print("exception", str(e))

		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('img', img)
	cv2.imshow('emotion probabilities', canvas)

	key = cv2.waitKey(1)

	if key & 0xFF == ord('a'):
		print("send message..")
		command = "face"
		json_str = '[{"command": "' + command + \
				   '", "emotion": "' + primary_emotion + '"}]\r\n'
		print(json_str)
		conn.send(json_str.encode('utf-8'))

	elif key & 0xFF == ord('q'):  #press q to quit
		break
	
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
