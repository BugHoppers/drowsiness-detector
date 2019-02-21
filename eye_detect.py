import cv2
import dlib
import numpy as np
from imutils import face_utils

# Function for detecting face from image and return the co-ordinates


def detectFace(img):
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    if face.empty():
        return []

    f_points = face.detectMultiScale(
        img, scaleFactor=1.3, minNeighbors=1, minSize=(20, 20))

    if len(f_points) == 0:
        return []

    f_points[:, 2:] += f_points[:, :2]
    return f_points


def eyesCrop(imgFrame):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    grayImg = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)

    faceList = detectFace(grayImg, minimumFeatureSize=(80, 80))

    if(len(faceList)) == 0:
        return None
    elif(len(faceList)) == 1:
        [face] = faceList
    elif(len(faceList)) > 1:
        face = faceList[0]

    rect_face = dlib.rectangle(left = int(face[0]), top = int(face[1]),
							    right = int(face[2]), bottom = int(face[3]))

    face_shape = predictor(grayImg, rect_face)
    face_shape = face_utils.shape_to_np(face_shape)

    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = face_shape[lBegin:lEnd]
    rightEye = face_shape[rBegin:rEnd]

    l_upperY = min(leftEye[1:3,1])
    l_lowY = max(leftEye[4:,1])
    l_difY = abs(l_upperY - l_lowY)

    lw = (leftEye[3][0] - leftEye[0][0])

    minxl = (leftEye[0][0] - ((34-lw)/2))
    maxxl = (leftEye[3][0] + ((34-lw)/2)) 
    minyl = (l_upperY - ((26-l_difY)/2))
    maxyl = (l_lowY + ((26-l_difY)/2))

    rect_left_eye = np.rint([minxl, minyl, maxxl, maxyl])
    rect_left_eye = rect_left_eye.astype(int)
    image_left_eye = grayImg[(rect_left_eye[1]):rect_left_eye[3], (rect_left_eye[0]):rect_left_eye[2]]


