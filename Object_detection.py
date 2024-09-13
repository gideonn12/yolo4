import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weight_path=r"C:\\Users\\gideo\\Desktop\\yolo4\\dnn_model\\yolov4.weights", cfg_path=r"C:\\Users\\gideo\\Desktop\\yolo4\\dnn_model\\yolov4.cfg"):
        print("Loading YOLOv4 weights and cfg files...")
        print("Running YOLOv4...")
        self.nms_threshold = 0.4
        self.cong_threshold = 0.5
        self.image_size = 608
        net = cv2.dnn.readNet(weight_path, cfg_path)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.model = cv2.dnn_DetectionModel(net)
        
        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, class_path=r"C:\\Users\\gideo\\Desktop\\yolo4\\dnn_model\\classes.txt"):
        with open(class_path, 'r') as f:
            self.classes = f.read().splitlines()
        self.color = np.random.uniform(0, 255, size=(80, 3))
        return self.classes
        
    def detect(self, frame):
        return self.model.detect(frame, self.nms_threshold, self.cong_threshold)