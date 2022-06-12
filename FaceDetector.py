import cv2
from numpy import sqrt

class FaceDetector_cv2():
    def __init__(self, scaleFactor=1.1, minNeighbors=10, maxPixels = 150000):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.maxPixels = maxPixels

    #input gray scale image
    #output a list of face region in the form of (x, y, w, h)
    def detect(self, img):
        height, width, _ = img.shape
        pixelCount = height * width
        if pixelCount > self.maxPixels:
            scale = sqrt(self.maxPixels / pixelCount)
            img = cv2.resize(img, (0, 0), fx = scale, fy = scale)
        else:
            scale = 1
        results = self.face_cascade.detectMultiScale(img, self.scaleFactor, self.minNeighbors)

        faces = []
        for (x, y, w, h) in results:
            scaleBack = 1 / scale
            faces.append((int(x * scaleBack), int(y * scaleBack), int(w * scaleBack), int(h * scaleBack)))
        return faces


#import mediapipe as mp

class FaceDetector_mediapipe():
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

    #input gray scale image
    #output a list of face region in the form of (x, y, w, h)
    def detect(self, img):
        results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []
        else:
            imgHeight, imgWidth, _ = img.shape
            
            faces = []
            for detection in results.detections:
                left = int(detection.location_data.relative_bounding_box.xmin * imgWidth)
                top = int(detection.location_data.relative_bounding_box.ymin * imgHeight)
                width = int(detection.location_data.relative_bounding_box.width * imgWidth)
                height = int(detection.location_data.relative_bounding_box.height * imgHeight)
                faces.append((left, top, width, height))
            return faces

from facenet_pytorch import MTCNN
import torch

class FaceDetector_mtcnn():
        def __init__(self):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.face_detection = MTCNN(keep_all=True, device=device)

        def detect(self, img):
            print(img.shape)
            if len(img) == 0 : return []
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results, _ = self.face_detection.detect(img_rgb)
            if results is not None : 
                faces = []
                for box in zip(results):
                    x = int(box[0][0])
                    y = int(box[0][1])
                    w = int(box[0][2])
                    h = int(box[0][3])
                    faces.append((x-10, y-10, w-x+20, h-y+20))
                return faces
            else : return []
