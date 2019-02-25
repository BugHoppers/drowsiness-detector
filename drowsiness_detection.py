import cv2
from eye_detect import eyesCrop
from keras.models import load_model
from model import cnnPreprocess
from imutils import resize

def main():
    cam = cv2.VideoCapture(0)
    model = load_model('eyeblink.hdf5')

    # blink_count counts the total number of blinks
    # close_count counts the consecutive close predictions
    # mem_count is the counter for the previous loop
    close_count = blink_count = mem_count = 0
    state = ''

    while True:
        _, frame = cam.read()
        frame = resize(frame, 400)

        # f_points = detectFace(frame)
        # print(f_points, sep='\n')
        
        eyes = eyesCrop(frame)
        if eyes is None:
            continue
        else:
            left_eye, right_eye = eyes

        prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0
        # print(prediction)
        
        # blinks
		# if the eyes are open reset the counter for close eyes
        if prediction > 0.5:
            state = 'Open'
            close_count = 0
        else:
            state = 'Closed'
            close_count += 1

		# if the eyes are open and previously were closed
		# for sufficient no. of frames then increcement the total blinks
        if state == 'Open' and mem_count > 1:
            blink_count += 1
		# keep the counter for the next loop 
        mem_count = close_count

		# draw the total number of blinks on the frame along with
		# the state for the frame
        cv2.putText(frame, "Blinks: {}".format(blink_count), (10, 30),
			cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Eyes: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 2)
		
        cv2.imshow('Drowsiness Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
