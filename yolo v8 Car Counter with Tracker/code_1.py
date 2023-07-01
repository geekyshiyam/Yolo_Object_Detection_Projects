from ultralytics import YOLO
import cv2
import math
import torch
from sort import *
import numpy as np
#print(torch.backends.mps.is.available())

model = YOLO('weights/yolov8n.pt')
cap = cv2.VideoCapture('cars.mp4')
mask = cv2.imread('mask.png')

tracker = Sort(max_age= 20, min_hits= 3, iou_threshold= 0.3)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"]
while True:
    success,img = cap.read()
    #cv2.imwrite('out.jpg',img)
    region = cv2.bitwise_and(img,mask)
    results = model(region, stream = True,device="mps")
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int (y2)
            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])
            curCls = cls
            if curCls == 2 and conf > 0.8:
                curAry = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,curAry))
    
    outTrackers = tracker.update(detections)
    for out in outTrackers:
        x1,y1,x2,y2,id = out
        x1,y1,x2,y2,id = int(x1), int(y1), int(x2), int (y2), int(id)
        #print(out)
        cv2.rectangle(img,(x1, y1),(x2, y2), (255,255,0), 1)
        xx1 = int(x2/2)    
        cv2.putText(img,f'{id}',(max(0,x1),max(50,y1)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
    

    cv2.imshow("image",img)
    #cv2.imshow("image",region)
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break
cv2.destroyAllWindows()

'''
Draw a boundary line
find the centre point of the bbox
check if the centre point touches the line
if it touches and the list doesnt have the id add it to list
total length of the list will the total count
'''
