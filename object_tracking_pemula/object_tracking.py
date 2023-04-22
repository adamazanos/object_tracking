# tutorial dari https://www.youtube.com/watch?v=O3b8lVF93jU
import cv2
from tracker import *

# Buat tracker object
tracker = EuclideanDistTracker()

# Capture video yang mau dideteksi

cap = cv2.VideoCapture('highway.mp4')

# Object detection dari kamera, Parameter bisa di ubah ubah
object_detector = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=20,detectShadows=True) 

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # Cara menghitung frame Box
    # [kordinat y awal : kordinat y akhir, kordinat x awal : kordinat x akhir]
    roi = frame[150: 360,300: 640]
    # 1. Deteksi Objek
    mask = object_detector.apply(roi)
    _, mask =cv2.threshold(mask,254, 255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])
    # 2. Tracking Object untuk dihitung
    boxes_ids =  tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # Tambah teks untuk menghitung object
        cv2.putText(roi, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0,0), 1)
        cv2.rectangle(roi, (x,y), (x + w, y + h), (0, 255, 0), 3)

    # Output berupa video dengan 3 output
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

