import cv2
import numpy
from Object_detection import ObjectDetection

ob = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")
cnt = 0

while True:
    a , frame = cap.read()
    (class_id, scores, boxes) = ob.detect(frame)
    for box in boxes:
        (x , y , w , h) = box
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 2)
        cnt += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()