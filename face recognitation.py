
# coding: utf-8

# In[2]:


import cv2
import os
import numpy as np


# In[11]:


# Face Recognition
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id=input('Enter ID ')
video_capture = cv2.VideoCapture(0)
num=0
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        num+=1
        cv2.imwrite("training-data/user."+id+'.'+str(num)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.waitKey(100)
    cv2.imshow('Video', canvas)
    cv2.waitKey(100)
    if num>20:
        break
video_capture.release()
cv2.destroyAllWindows()


# ## Training

# In[2]:


from PIL import Image


# In[8]:


recognizer = cv2.face.LBPHFaceRecognizer_create()
path="training-data"


# In[10]:


def getImagesAndLabels(path):
    faces=[]
    Ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    for imagePath in imagePaths:
        faceImage=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        Ids.append(Id)
    return Ids,faces
Ids,faces=getImagesAndLabels(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainfacedata.yml')


# ## Detector

# In[18]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainfacedata.yml")
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 0, 255)
id=0
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        if Id==1:
            Id="Himanshu"
        else:
            
        cv2.putText(frame,str(Id),(x,y+h),font,fontscale,fontcolor)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

