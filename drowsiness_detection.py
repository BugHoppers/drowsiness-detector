import cv2
from eye_detect import eyesCrop
from keras.models import load_model
from model import cnnPreprocess
from imutils import resize


def main():
    cam = cv2.VideoCapture(0)
    model = load_model('eyeblink.hdf5')

    close_count = 0
    state = ''

    while True:
        _, frame = cam.read()
        frame = resize(frame, 500)

        eyes = eyesCrop(frame)
        if eyes is None:
            continue
        else:
            eye1, eye2 = eyes

        if eye2 is None:
            cv2.imshow('Eye1', eye1)
            prediction = model.predict(cnnPreprocess(eye1))

        else:
            cv2.imshow('Eye2', eye2)
            prediction = (model.predict(cnnPreprocess(eye1)) +
                          model.predict(cnnPreprocess(eye2)))/2.0

        if prediction > 0.5:
            state = 'Open'
            close_count = 0
        else:
            state = 'Closed'
            close_count += 1

        cv2.putText(frame, "Blinks: {}".format(close_count), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Eyes: {}".format(state), (300, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 2)

        if close_count > 10:
            cv2.putText(frame, "ALERT", (100, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 2)

        cv2.imshow('Drowsiness Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
