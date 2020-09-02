# USAGE
# python detect_age_video.py --face face_detector --age age_detector

# import the necessary packages
import collections
import json
from operator import itemgetter

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

def save_and_clearList(lista):
	"""ref.push({
		'elemento': json.dumps(str(lista[0]))
	})"""
	lista.clear()


def  ageAndGenderPredict(frame, faceNet, ageNet,genderNet, minConf=0.5):
	# define the list of age buckets our age detector will predict
	ageList = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]
	genderList = ['M', 'F']
	# initialize our results list
	results = []
	faces=[]
	prev_Faces=[]



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
		faceID=i
		prev_Faces.append(faceID)
		for k,j in zip(prev_Faces,faces):
			if prev_Faces[j]==faces[i]:
				continue
			else:
				faceID+=1
				prev_Faces.append(faceID)


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
			"""ageNet.setInput(faceBlob)
			agePreds = ageNet.forward()
			age = ageList[agePreds[0].argmax()]

			genderNet.setInput(faceBlob)
			genderPreds = genderNet.forward()
			gender = genderList[genderPreds[0].argmax()]"""
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = ageList[i]
			ageConfidence = preds[0][i]

			genderNet.setInput(faceBlob)
			genderPreds = genderNet.forward()
			j = genderPreds[0].argmax()
			gender = genderList[j]
			genderConfidence = genderPreds[0][j]
			# construct a dictionary consisting of both the face
			# bounding box location along with the age prediction,
			# then update our results list
			"""d = {
				"loc": (startX, startY, endX, endY),
				"age": age,
				"gender": gender,
				"ID":faceID
			}"""
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age),
				"ageConf":(ageConfidence),
				"gender": (gender),
				"genderConf": (genderConfidence),
				"ID": faceID
			}
			results.append(d)


	# return our results to the calling function
	return results

# Fetch the service account key JSON file contents
"""cred = credentials.Certificate('sistema-sicurezza-con-alexa-firebase-adminsdk-zf6l0-4c89f84297.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sistema-sicurezza-con-alexa.firebaseio.com/'
})"""
cred = credentials.Certificate('myprojectalexa-df864-firebase-adminsdk-q4l0k-03573c4c36.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://myprojectalexa-df864.firebaseio.com/'
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
vs = VideoStream(src=0).start()




faceIdList=[]
ageList=[]
genderList=[]
lista=[]
numSec = 0
# loop over the frames from the video stream
while True:

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame, and for each face in the frame,
	# predict the age
	results = ageAndGenderPredict(frame, faceNet, ageNet,genderNet,
		minConf=defaultConfidence)
	age = ""
	gender = ""
	# loop over the results
	for r in results:
		age = r["age"]
		gender = r["gender"]
		ID = r["ID"]
		lista.append(r)
		# draw the bounding box of the face along with the associated
		# predicted age
		text = "{}{}{}".format(age, gender,ID)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	count = len([elem for elem in results if elem["ID"]==0])
	#print("Massimo",count)
	#Save
	ref = db.reference('/persone')
	current_time = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")
	#indice con percentuale più alta
	size = len(lista)
	f = itemgetter("ageConf")
	lista.sort(key=f, reverse=True)
	#print("Ordered list", lista)
	#print("Massimo", lista[0])
	#Indice più frequente (riferito all'età)
	[(occorrenze,_)] = collections.Counter(x['age'] for x in lista).most_common(1)
	print("Occorrenze con counter",occorrenze)

	"""if age and gender:
		print("Salvo su db")
		ref.push({
			'elemento': json.dumps(str(lista[0]))
		})
	if(numSec%100 == 0):
		if results:
			current_time = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")

	numSec+=1
"""
	if (numSec % 100 == 0):
		if results:
			current_time = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")
	numSec+=1

	if not lista:
		print("Non ci sono elementi da salvare")
	else:
		print("Invoco e salvo")
		print("15 sec",numSec)
		print("Lunghezza lista",len(lista))
		print("Lista",lista)
		#print("Elemento 0{}elemento 1".format(lista[0],lista[1]))
		if numSec==15:
			save_and_clearList(lista)
			numSec=0


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()