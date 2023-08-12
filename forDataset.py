import cv2
import numpy as np

cap = cv2.VideoCapture(0)

count = 0

while cap.isOpened():
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.imwrite("savedImages/frame%d.jpg" % count, frame)

    count += 1

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()