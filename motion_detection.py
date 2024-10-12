import cv2
from pathlib import Path
import numpy as np
import time

def create_rg_mask(cur_gray_frame, prev_gray_frame):
    diff_frame = cv2.absdiff(cur_gray_frame, prev_gray_frame)
    _, thresh = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    motion_frame = np.zeros_like(frame)
    motion_frame[thresh == 0] = [0, 255, 0]
    cv2.drawContours(motion_frame, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
    res = cv2.addWeighted(motion_frame, 0.8, frame, 1, 0)
    return res

def add_text(frame, text):
    frame = cv2.putText(frame, org=(50, 50), text=text, color=(0,0,0), fontFace=cv2.LINE_AA,\
            fontScale=1, thickness=3)
    frame = cv2.putText(frame, org=(300, 50), text=str(int(TIMER-(time.time()-start_time))),\
            color=(0,0,0), fontFace=cv2.LINE_AA, fontScale=1, thickness=2)
    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, prev_gray_frame = cap.read()
prev_gray_frame = cv2.cvtColor(prev_gray_frame, cv2.COLOR_BGR2GRAY)
prev_gray_frame = cv2.GaussianBlur(prev_gray_frame, (25, 25), 0)
start_time = time.time()
detect_motion = True
TIMER = 10

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame.")
        break
    if time.time() - start_time > TIMER:
        detect_motion = not(detect_motion)
        start_time = time.time()
    
    cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cur_gray_frame = cv2.GaussianBlur(cur_gray_frame, (25, 25), 0)

    if detect_motion:
        diff_frame = cv2.absdiff(cur_gray_frame, prev_gray_frame)
        if np.any((np.any(diff_frame, axis=0)), axis=0) == False:
            continue
        res = create_rg_mask(cur_gray_frame, prev_gray_frame)
        
        frame = add_text(frame, "Red light")
        cv2.imshow('Motion detection', np.concatenate((frame, res), axis=0))
    else:
        first_frame = frame.copy()
        first_frame = add_text(first_frame, "Green light")

        cv2.imshow('Motion detection', np.concatenate((first_frame, frame), axis=0))
    prev_gray_frame = cur_gray_frame.copy()
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()