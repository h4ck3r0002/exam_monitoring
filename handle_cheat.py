import cv2
import sys
import os
import django
import numpy as np
from django.conf import settings
from collections import deque 
import joblib 
from skimage.feature import hog

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
django.setup()

from quiz.models.quiz import Monitor, Result 
from modules.SCRFD import SCRFD


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


onnxmodel = "scrfd_500m_kps.onnx"
confThreshold = 0.5
nmsThreshold = 0.5
mynet = SCRFD(onnxmodel)


def process_video(monitor_id):
    reason = ''

    # Lấy đối tượng Monitor
    monitor = Monitor.objects.get(id=monitor_id)
    video_path = monitor.video.path

    # Mở video
    camera = cv2.VideoCapture(video_path)
    is_cheat = False

    count_fraud = 0
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
        if len(scores)>1:
            count_fraud+=1
        else:
            if bboxes.shape[0] > 0 or lmarks.shape[0] > 0:

                frame = visualize(frame, bboxes, lmarks, scores, fps=tm.getFPS())

                # Check if all coordinates of the highest score face in the frame
                position = "normal"
                roll, yaw, pitch = find_pose(lmarks[0])
                if yaw > 40:
                    position = "trai"
                    # threading.Thread(target=play_audio, args=("amthanh/trai.mp3",)).start()
                    count_fraud += 1
                elif yaw < -40:
                    position = "phai"
                    # threading.Thread(target=play_audio, args=("amthanh/phai.mp3",)).start()
                    count_fraud += 1
                if pitch > 25:
                    position = "tren"
                    # threading.Thread(target=play_audio, args=("amthanh/tren.mp3",)).start()
                    count_fraud += 1
                lmarks = lmarks.astype(int)
                start_point = (lmarks[0][2][0], lmarks[0][2][1])
                end_point = (lmarks[0][2][0] - int(yaw), lmarks[0][2][1] - int(pitch))

                cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)
                bn = "\n"
                cv2.putText(
                    frame,
                    f"{position}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    thickness=1,
                )
            else:
                # threading.Thread(
                #     target=play_audio, args=("amthanh/khongphathienkhuonmat.mp3",)
                # ).start()
                cv2.putText(
                    frame,
                    f"khong phat hien khuon mat",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    thickness=1,
                )
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

    if count_fraud > 10:
        is_cheat = True
        
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