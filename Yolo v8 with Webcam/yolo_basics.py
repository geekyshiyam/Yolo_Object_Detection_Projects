from ultralytics import YOLO
import cv2
import math
import torch
#print(torch.backends.mps.is.available())

model = YOLO('weights/yolov8l.pt')
cap = cv2.VideoCapture('office_360.mp4')
cap.set(3,640)
cap.set(4,480)

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
    results = model(img, stream = True,device="mps")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int (y2)
            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])
            cv2.rectangle(img,(x1, y1),(x2, y2), (255,255,0), 3)
            xx1 = int(x2/2)
            cv2.putText(img,f'{classNames[cls]}',(max(x1,0),max(y2,50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
            cv2.putText(img,f'{conf}',(max(0,x1),max(50,y1)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.imshow("image",img)
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break
cv2.destroyAllWindows()