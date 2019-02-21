import cv2
from eye_detect import detectFace

def main():
    cam = cv2.VideoCapture(0)

    while True :
        _, frame = cam.read()

        cv2.imshow('Drowsiness Detector', frame)

        f_points = detectFace(frame)
        # print(f_points, sep='\n')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == '__main__':
	main()