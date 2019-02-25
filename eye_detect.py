import cv2

def eyesCrop(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            return None
        elif len(eyes) == 1 :
            ex,ey,ew,eh = eyes[0]
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
            return eye, None
        else :
            ex,ey,ew,eh = eyes[0]
            eye1 = roi_gray[ey:ey+eh, ex:ex+ew]
            eye1 = cv2.resize(eye1, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
            ex,ey,ew,eh = eyes[1]
            eye2 = roi_gray[ey:ey+eh, ex:ex+ew]
            eye2 = cv2.resize(eye2, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)

            return eye1, eye2