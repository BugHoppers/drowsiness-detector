import cv2

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
