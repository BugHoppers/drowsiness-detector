import cv2
from eye_detect import eyesCrop


def main():
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()

        # f_points = detectFace(frame)
        # print(f_points, sep='\n')
        
        eyes = eyesCrop(frame)
        if eyes is None:
            continue
        else:
            left_eye, right_eye = eyes

        cv2.imshow('Drowsiness Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
