import cv2
import numpy as np
from Object_detection import ObjectDetection

# Initialize object detection
ob = ObjectDetection()

# Capture video
cap = cv2.VideoCapture("germany.mp4")
total_cars = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    (class_id, scores, boxes) = ob.detect(frame)
    car_count = 0


    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
    
    # Draw bounding boxes and count cars
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        car_count += 1

    # Update the total car count
    total_cars += car_count

    # Display the car count on the frame
    cv2.putText(frame, f"Cars in frame: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)    
    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break
    # Check if the window was closed
    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
