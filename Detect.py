import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Load the model we trained
model = load_model('Mask_mobilenetmodel.keras')

#load the haarcascade to detect the faces
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)
# for detecting the faces
def dnn_face_model(Frame):
    h,w = Frame.shape[:2]

    blob = cv2.dnn.blobFromImage(Frame,1.0,(300,300),(104.0,177.0,123.0))

    net.setInput(blob)

    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7]*[w,h,w,h]
            x1,y1,x2,y2 = box.astype(int)
            faces.append((x1,y1,x2-x1,y2-y1))

    return faces

# predicting the mask or no mask to face
def predict_img(face):
    face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    face = cv2.resize(face,(224,224))
    face = face / 255.0
    face = face.reshape(1,224,224,3)

    print(len(face))

    prediction = model.predict(face)

    confidence = prediction[0][0]

    print("Prediction:", prediction[0][0])


    if confidence > 0.5:
        label = f'No Mask ({confidence*100:.2f}%)'
        color =(0,0,255)
    else:
        label = f'Mask ({(1-confidence)*100:.2f}%)'
        color = (0,255,0)

    return label,color
    
#input in image mode
def image_mode():
    path = input('Enter the image path:').strip().replace('"','')
    img = cv2.imread(path)

    if img is None:
        print('image is not found')
        return
    
    
    faces = dnn_face_model(img)
    print("Faces detected:", len(faces))

    for (x,y,w,h) in faces:
        face = img[y:y+h,x:x+w]

        if face.size == 0:
            continue
        
        label,color = predict_img(face)

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        y_text = max(y-10, 20)

        cv2.putText(img, label, (x,y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
       

    cv2.imshow('image:',img)
    cv2.imshow('ROI:',face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#input as web cam mode
def webcam_mode():
    cap = cv2.VideoCapture(0)

    while True:
        re,Frame = cap.read()
        if not re:
            break
        faces = dnn_face_model(Frame)
        print("Faces detected:", len(faces))
        

        for (x,y,w,h) in faces:
            roi = Frame[y:y+h,x:x+w]

            if roi.size == 0:
                continue

            label,color = predict_img(roi)
            cv2.rectangle(Frame,(x,y),(x+w,y+h),color,2)
            y_text = max(y-10, 20)

            cv2.putText(Frame, label, (x,y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('live mask detection',Frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows() 

print('user choice 1 if image 2 if web cam')

choise = int(input('enter your choise:'))

if choise == 1:
    image_mode()
elif choise == 2:
    webcam_mode()
else:
    print('invalid choice')











