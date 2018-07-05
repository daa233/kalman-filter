# Mouse Tracking with Kalman Filter
# Author: Du Ang
# Material references: https://blog.csdn.net/angelfish91/article/details/61768575
# Date: July 1, 2018

import cv2
import numpy as np

from kalman_filter import KalmanFilter

TITLE = "Mouse Tracking with Kalman Filter"
frame = np.ones((800,800,3),np.uint8) * 255


def mousemove(event, x, y, s, p):
    global frame, current_measurement, current_prediction
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    current_prediction = kalman.predict()

    cmx, cmy = current_measurement[0], current_measurement[1]
    cpx, cpy = current_prediction[0], current_prediction[1]

    frame = np.ones((800,800,3),np.uint8) * 255
    cv2.putText(frame, "Measurement: ({:.1f}, {:.1f})".format(np.float(cmx), np.float(cmy)),
                (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
    cv2.putText(frame, "Prediction: ({:.1f}, {:.1f})".format(np.float(cpx), np.float(cpy)),
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
    cv2.circle(frame, (cmx, cmy), 10, (50, 150, 0), -1)      # current measured point
    cv2.circle(frame, (cpx, cpy), 10, (0, 0, 255), -1)      # current predicted point

    kalman.correct(current_measurement)

    return


cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mousemove)

stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
estimateCovariance = np.eye(stateMatrix.shape[0])
transitionMatrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
measurementStateMatrix = np.zeros((2, 1), np.float32)
observationMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
kalman = KalmanFilter(X=stateMatrix,
                      P=estimateCovariance,
                      F=transitionMatrix,
                      Q=processNoiseCov,
                      Z=measurementStateMatrix,
                      H=observationMatrix,
                      R=measurementNoiseCov)

while True:
    cv2.imshow(TITLE,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
