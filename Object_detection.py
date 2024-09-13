import cv2
import numpy

class ObjectDetection():
    def __init__(self, weight_path=r'C:\Users\gideo\Desktop\objdect\dnn_model\yolov4.weights', cfg_path=r'C:\Users\gideo\Desktop\objdect\dnn_model\yolov4.cfg'):
        print("Loading YOLOv4 weights and cfg files...")
        print("Running YOLOv4...")
        self.nms_threshold = 0.4
        self.cong_threshold = 0.5
        self.image_size = 608
        net = cv2.dnn.readNet(weight_path, cfg_path)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)	
        