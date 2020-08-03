#A Gender and Age Detection program by Mahesh Sawant

import cv2
import numpy as np
import math
import argparse
from datetime import datetime
import time
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
from pyimagesearch.centroidtracker import CentroidTracker

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()   #copio frame in un altra variabile
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #1.0 scalefactor ci potrebbe permettere di ridimensionare le immagini, con 1.0 le lasciamo intatte
    #(300,300) dimensioni spaziale che la rete neurale si aspetta
    #[104,117,123] valori della sottrazione media
    #true opencv assume che le immagini siano in BGR, il valore mean presuppone che stiamo usando l'ordine RGB. Per risolvere
    # questa discrepanza possiamo scambiare i canali R e B nell'immagine impostando TRUE

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def connectionAndInsert(id,sex,age,timee):

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='Persone',
                                             user='root',
                                             password='root')
        cursor = connection.cursor()
        mySql_insert_query = """INSERT INTO ingressi (id,sesso,eta,orario) 
                               VALUES 
                               (%s, %s, %s, %s) """

        recordTuple = (id, sex, age,timee)
        cursor.execute(mySql_insert_query, recordTuple)
        connection.commit()
        #cursor = connection.cursor()
        #cursor.execute(mySql_insert_query)
        #connection.commit()
        print(cursor.rowcount, "Record inserted successfully into ingressi table")
        cursor.close()

    except mysql.connector.Error as error:
        print("Failed to insert record into ingressi table {}".format(error))

    finally:
        if (connection.is_connected()):
            connection.close()
            print("MySQL connection is closed")

def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)


insert = False
parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"



MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

#Prove

i=0
id=0
t0 = time.time()
while cv2.waitKey(5000)<0:
    hasFrame,frame=video.read()
    """cv2.imwrite('frame' + str(i) + '.jpg', frame)
    #i += 1
    #image1 = cv2.imread("frame5.jpg")
    #image2 = cv2.imread("frame5.jpg")

    difference = cv2.subtract(image1,image2)

    result = not np.any(difference)
    if result:
        print("The image are the same")
    else:
        print("Differenti")
"""
    t1=time.time()
    num_sec = t1-t0
    if num_sec>30:
        #print("Inserimento dati a tempo scaduto")
        #connectionAndInsert()
        break
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")



    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]




        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        ageD=[age]
        gendD=[gender]

        now = datetime.now()

        currentTime = now.strftime("%H:%M:%S")
        print(age,gendD,currentTime)

        #print("Inserimento dati a tempo scaduto")
        #stringa = convert_list_to_string(age,'')
        #etab=stringa.split('-')[0]
        #eta = etab.strip('(')
        #stringaGenere = convert_list_to_string(gender,)
        #print("Eta",eta)
        #connectionAndInsert(id, gendD[0],eta, now)

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)





