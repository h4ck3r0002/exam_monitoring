import cv2
import sys
import os
import django
import numpy as np
from django.conf import settings
from collections import deque
import threading
import pygame
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "exam_monitoring.settings")
django.setup()

from quiz.models.quiz import Monitor, Result


import cv2
import numpy as np
from modules.SCRFD import SCRFD


def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    pygame.time.set_timer(pygame.USEREVENT, 3000)


def visualize(image, boxes, lmarks, scores, fps=0):
    for i in range(len(boxes)):
        print(boxes[i])
        xmin, ymin, xmax, ymax, score = boxes[i].astype("int")
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        for j in range(5):
            cv2.circle(
                image,
                (int(lmarks[i, j, 0]), int(lmarks[i, j, 1])),
                1,
                (0, 255, 0),
                thickness=-1,
            )
    return image


def are_coordinates_in_frame(frame, box, pts):

    height, width = frame.shape[:2]

    if np.any(box <= 0) or np.any(box >= height) or np.any(box >= width):
        return False
    if np.any(pts <= 0) or np.any(pts >= height) or np.any(pts >= width):
        return False

    return True


def find_pose(points):
    LMx = points[:, 0]  # points[0:5]# horizontal coordinates of landmarks
    LMy = points[:, 1]  # [5:10]# vertical coordinates of landmarks

    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = LMy[1] - LMy[0]
    angle = np.arctan(dPy_eyes / dPx_eyes)  # angle for rotation based on slope

    alpha = np.cos(angle)
    beta = np.sin(angle)

    # rotated landmarks
    LMxr = alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2
    LMyr = -beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2

    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2

    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2

    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal


onnxmodel = "models/scrfd_500m_kps.onnx"
confThreshold = 0.5
nmsThreshold = 0.5
mynet = SCRFD(onnxmodel)

count_face = 0
count_fraud = 0
is_cheat = False
def process_video(monitor_id):
    reason = ""
    global count_face
    global count_fraud
    global is_cheat

    # Lấy đối tượng Monitor
    monitor = Monitor.objects.get(id=monitor_id)
    video_path = monitor.video.path

    # Mở video
    camera = cv2.VideoCapture(video_path)
    is_cheat = False

    frame_count = 0
    start_time = time.time()
    tm = cv2.TickMeter()
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        tm.start()  # for calculating FPS
        bboxes, lmarks, scores = mynet.detect(frame)  # face detection
        tm.stop()
        
        if len(bboxes)>1:
            count_face+=1
        elif len(bboxes)==0:
            count_fraud += 1
        # Tính toán FPS
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # Làm tròn FPS về số nguyên
        fps_int = int(round(fps))
        

        
        # Vẽ FPS lên khung hình
        cv2.putText(
            frame,
            f"FPS: {fps_int}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if count_fraud > 50:
        is_cheat = True
    if count_face>10:
        is_cheat= True
    camera.release()

    # Cập nhật trạng thái gian lận trong cơ sở dữ liệu
    monitor.is_cheat = is_cheat
    monitor.reason = reason
    monitor.save()

    print(monitor.exam)
    print(monitor.user)

    result = Result.objects.get(exam=monitor.exam, user=monitor.user)
    result.is_cheat = is_cheat
    result.reason = reason
    result.is_done = True
    result.save()


if __name__ == "__main__":
    monitor_id = int(sys.argv[1])
    process_video(monitor_id)
