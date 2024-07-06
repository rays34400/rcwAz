import cv2,numpy as np,face_recognition
 
 
 
from fastapi import FastAPI

# Création d'une instance de l'application FastAPI
app = FastAPI()

# Définir des routes avec des opérations
@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

#import_Signatures
signatures_class=np.load('FaceSignature_db.npy')
X=signatures_class[ : , 0: -1].astype('float')
Y=signatures_class[ : , -1]
 
#open camera
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    if success:
        print('capturing...')
        imgR=cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgR=cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)
       
        #find face location from the webcam
        facesCurrent = face_recognition.face_locations(imgR)
       
        #get Signatures from faces
        encodesCurrent = face_recognition.face_encodings(imgR, facesCurrent)
        for encodeFace , faceloc in zip(encodesCurrent,facesCurrent):
            matches=face_recognition.compare_faces(X, encodeFace)
            faceDis =face_recognition.face_distance(X, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name=Y[matchIndex].upper()
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255), 2)
                cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,0,255), cv2.FILLED)
                cv2.putText(img,name,(x1+10,y2-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            else:
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255), 2)
                cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,0,255), cv2.FILLED)
                cv2.putText(img,'unknown',(x1+10,y2-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
               
        cv2.imshow('webcam',img)
        cv2.waitKey(1)