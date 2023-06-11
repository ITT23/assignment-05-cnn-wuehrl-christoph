# This program loads a model trained on the hagrid dataset and controls media players
# Because I have no white wall it only works with the cardboard with aruco markers we got in the course
# This board should point with the smaller side to the ground to have more space for your hand
# To avoid false input only input in front of the cardboard is accepted (you camera picture is shown to see if the aruco detection works)
# The gestures are like (volume up), dislike (volume down) and rock (start and stop the player)
from pynput.keyboard import Key, Controller
import keras
import cv2
import cv2.aruco as aruco
import numpy as np
from time import sleep

KEYBOARD = Controller()
model = None
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3
LABELS = ['like', 'no_gesture', 'dislike', 'rock']
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WAIT_TIME_AFTER_GESTURE_RECOGNITION = 0.6

def main():
    load_model()
    capture_video()

def capture_video():
    cap = cv2.VideoCapture(0)
    while(True):
        found_markers = False
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_params = aruco.DetectorParameters()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        # The method to detect markers and transform the image is form the AR-Game in the grips course
        if ids is not None:
            # only proceed if four markers are detected
            if len(ids) == 4:
                # find inner corner of each marker and apply perspective transformation
                
                top_left = corners[list(ids).index(0)][0][2]
                bottom_left = corners[list(ids).index(1)][0][3]
                bottom_right = corners[list(ids).index(2)][0][0]
                top_right = corners[list(ids).index(3)][0][1]

                pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])

                x0 = 0
                y0 = 0
                x1 = WINDOW_WIDTH
                y1 = WINDOW_HEIGHT
                FLIP_IMAGE = True

                if FLIP_IMAGE:
                    pts2 = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
                else:
                    pts2 = np.float32([[x1, y1], [x0, y1], [x1, y0], [y0, y0]])

                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                frame = cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))
                found_markers = True
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # to quite the program
            break
        resized = cv2.resize(frame[50:350, 60:210], SIZE)
        resized.shape
        reshaped = resized.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
        reshaped.shape
        prediction = model.predict(reshaped)
        label = LABELS[np.argmax(prediction)]
        if(label == 'rock' and found_markers):
            start_pause_track()
            sleep(WAIT_TIME_AFTER_GESTURE_RECOGNITION)
        elif(label == 'like' and found_markers):
            volume_up()
            sleep(WAIT_TIME_AFTER_GESTURE_RECOGNITION)
        elif(label == 'dislike' and found_markers):
            volume_down()
            sleep(WAIT_TIME_AFTER_GESTURE_RECOGNITION)
        


def load_model():
    global model
    model = keras.models.load_model("gesture_recognition")

def start_pause_track():
    KEYBOARD.press(Key.media_play_pause)
    KEYBOARD.release(Key.media_play_pause)

def volume_up():
    KEYBOARD.press(Key.media_volume_up)
    KEYBOARD.release(Key.media_volume_up)

def volume_down():
    KEYBOARD.press(Key.media_volume_down)
    KEYBOARD.release(Key.media_volume_down)
main()