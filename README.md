# yolo4

Object Detection Using YOLOv4 - README
Introduction
This project demonstrates real-time object detection in video using the YOLOv4 deep learning model. The code loads a pre-trained YOLOv4 model, applies object detection on frames of a video, and highlights detected objects, specifically focusing on counting the number of cars.

Features
Uses YOLOv4 for object detection.
Capable of counting cars in each frame of the video.
Draws bounding boxes around detected cars and displays the count of cars detected in the frame.
Non-Maximum Suppression (NMS) is used to eliminate overlapping bounding boxes.
Requirements
Ensure you have the following software and dependencies installed:

Python 3.x
OpenCV (cv2)
Numpy (numpy)
YOLOv4 weights, configuration file, and class names file
Installing Dependencies
Use the following command to install the required Python libraries:

bash
Code kopiëren
pip install opencv-python numpy
Directory Structure
Place your files in the following structure:

bash
Code kopiëren
project_folder/
│
├── dnn_model/
│   ├── yolov4.weights     # Pre-trained YOLOv4 weights
│   ├── yolov4.cfg         # YOLOv4 configuration file
│   └── classes.txt        # File containing object class names
│
├── germany.mp4            # Input video file (example: a video taken in Germany)
└── object_detection.py    # Python file containing the ObjectDetection class and main script
ObjectDetection Class
This class initializes and handles the YOLOv4 model, loads class names, and performs object detection on individual frames of a video.

Methods:
__init__: Initializes the model by loading the YOLOv4 weights, configuration, and class names. It sets parameters for Non-Maximum Suppression (NMS) and confidence threshold.
load_class_names: Loads the class names from a provided text file (e.g., classes.txt).
detect: Detects objects in a given frame and returns the detected class IDs, confidence scores, and bounding boxes.
Running the Object Detection
To run the object detection, follow these steps:

Download the YOLOv4 weight file (yolov4.weights), configuration file (yolov4.cfg), and class names file (classes.txt).
Place these files in the dnn_model folder as described in the directory structure.
Place the video file you want to process (e.g., germany.mp4) in the project folder.
Run the Python script:
bash
Code kopiëren
python object_detection.py
Code Explanation
The code initializes object detection using YOLOv4 and reads the video frame-by-frame:

Loading the Model:

The YOLOv4 model is loaded using the .weights and .cfg files.
The class labels are loaded from the classes.txt file.
Processing the Video:

A video is captured using OpenCV (cv2.VideoCapture).
Each frame of the video is processed to detect objects (specifically cars).
Non-Maximum Suppression:

NMS is applied to remove redundant bounding boxes and only keep the best one for each detected object.
Displaying Results:

The number of cars in each frame is displayed, and bounding boxes are drawn around the detected cars.
The total number of cars detected is updated as the video progresses.
Exiting the Program:

The video display can be exited by pressing the "ESC" key or if the window is closed manually.
