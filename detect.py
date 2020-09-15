# USAGE
# python detect_age_video.py --face face_detector --age age_detector

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from collections import Counter

def save(age,gender):
	ref = db.reference('/persone')
	current_time = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")
	if age and gender:
		print("Salvo su db")
		ref.push({
			'age': age,
			'gender': gender,
			'time': current_time
		})

def ageConf(val):
	return val["age"][1]


def genderConf(val):
	return val["gender"][1]


def  ageAndGenderPredict(frame, faceNet, ageNet,genderNet, minConf=0.5):
	# define the list of age buckets our age detector will predict

	# initialize our results list
	results = []

	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):

		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > minConf:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the ROI of the face
			face = frame[startY:endY, startX:endX]

			# ensure the face ROI is sufficiently large
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# construct a blob from *just* the face ROI
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# make predictions on the age and find the age bucket with
			# the largest corresponding probability
			ageNet.setInput(faceBlob)
			agePreds = ageNet.forward()

			age = ageList[agePreds[0].argmax()]
			ageConfidence = agePreds[0][agePreds[0].argmax()]

			genderNet.setInput(faceBlob)
			genderPreds = genderNet.forward()
			genderConfidence = genderPreds[0][genderPreds[0].argmax()]
			gender = genderList[genderPreds[0].argmax()]
			# construct a dictionary consisting of both the face
			# bounding box location along with the age prediction,
			# then update our results list
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age,ageConfidence),
				"gender": (gender,genderConfidence)
			}
			results.append(d)


	# return our results to the calling function
	return results

ageList = ["(0-3)", "(4-7)", "(8-14)","(15-24)","(25-37)",
		"(38-47)", "(48-59)", "(60-100)"]
genderList = ['M', 'F']
# Fetch the service account key JSON file contents
cred = credentials.Certificate('sistema-sicurezza-con-alexa-firebase-adminsdk-zf6l0-4c89f84297.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sistema-sicurezza-con-alexa.firebaseio.com/'
})

defaultConfidence = 0.5

print("Carico il face model...")
facePrototxtPath = "face_detector/deploy.prototxt"
faceWeightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(facePrototxtPath,faceWeightsPath)

print("Carico age model...")
agePrototxtPath = "age_detector/age_deploy.prototxt"
ageWeightsPath = "age_detector/age_net.caffemodel"
ageNet = cv2.dnn.readNet(agePrototxtPath,ageWeightsPath)


print("Carico gender model...")
genderPrototxtPath = "gender_detector/gender_deploy.prototxt"
genderWeightsPath = "gender_detector/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderPrototxtPath,genderWeightsPath)

print("Inizia lo stream...")
"""
vs = VideoStream('test/1brasile.mp4').start()
"""
cap = cv2.VideoCapture('test/37brasilelungo.mp4')

persone = []
numSec = 1
# loop over the frames from the video stream
"""while True:"""
while cap.isOpened():

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	"""frame = vs.read()"""
	ret,frame = cap.read()

	frame = imutils.resize(frame, width=400)

	# detect faces in the frame, and for each face in the frame,
	# predict the age

	results = ageAndGenderPredict(frame, faceNet, ageNet,genderNet,
	minConf=defaultConfidence)
	age = ""
	gender = ""

	# loop over the results
	for r in results:
		age = r["age"][0]
		ageConfidence = r["age"][1]*100

		gender = r["gender"][0]
		genderConfidence = r["gender"][1]*100

		(startX, startY, endX, endY) = r["loc"]
		persone.append(r)

		# draw the bounding box of the face along with the associated
		# predicted age
		textAge = "{}:{:.2f}%".format(age,ageConfidence)
		textGender = "{}:{:.2f}%".format(gender, genderConfidence)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(255, 255, 255), 1)
		cv2.putText(frame, textAge, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
		cv2.putText(frame, textGender, (startX, endY+20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

	if(numSec%120 == 0 ):
		if(persone):
			print("***********\nEuristica 1")
			[(eta,x)]=Counter(x["age"][0] for x in persone ).most_common(1)
			[(sesso,_)]=Counter(x["gender"][0] for x in persone ).most_common(1)
			print((eta,sesso))

			print("Euristica 2")
			l = list(persone)
			l.sort(key=ageConf,reverse = True)
			eta=l[0]["age"][0]
			l.sort(key=genderConf, reverse = True)
			sesso=l[0]["gender"][0]
			print((eta,sesso))

			print("Euristica 3")
			[(eta,_)] = Counter(x["age"][0] for x in persone if x and  x["age"][1]>0.7).most_common(1)
			[(sesso,_)] = Counter(x["gender"][0] for x in persone if x and   x["gender"][1]>0.7).most_common(1)
			print((eta, sesso))


			#Save
			save(eta,sesso)
			persone.clear()
		else:
			print("Nessun elemento da salvare")
	numSec+=1



	# show the output frame
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(10) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
"""vs.stop()"""
