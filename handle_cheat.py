import cv2
import sys
import os
import django
import numpy as np
from django.conf import settings
from collections import deque
import threading
import django
import numpy as np
from django.conf import settings
from collections import deque 
import joblib 
import os 
import mediapipe as mp
# from skimage.feature import hog

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
django.setup()

from quiz.models.quiz import Monitor, Result

# Tải mô hình, scaler và label encoder với tên file mới
model = joblib.load('cheat_detection_pose_model.pkl')
scaler = joblib.load('scaler_pose.pkl')
label_encoder = joblib.load('label_encoder_pose.pkl')

# Khởi tạo mô hình Pose của MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Hàm trích xuất các keypoints của pose
def extract_pose_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    else:
        return None 


def predict_image(frame):
    keypoints = extract_pose_keypoints(frame)
    if keypoints is not None:
        keypoints = keypoints.reshape(1, -1)  # Định dạng lại để phù hợp với đầu vào của mô hình
        keypoints = scaler.transform(keypoints)  # Chuẩn hóa

        # Dự đoán
        prediction = model.predict(keypoints)
        probabilities = model.decision_function(keypoints)  # Lấy giá trị quyết định
        probabilities = 1 / (1 + np.exp(-probabilities))  # Chuyển đổi thành xác suất

        return label_encoder.inverse_transform(prediction)[0], probabilities[0]
    else:
        return "No Pose Detected", None


def process_video(monitor_id):
    reason = ''

    # Mở video
    camera = cv2.VideoCapture(video_path)
    is_cheat = False
    count = 0 
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        # Dự đoán với từng khung hình
        label, prob = predict_image(frame)
        if label == "cheat":
            count += 1

        # Nếu phát hiện gian lận
        if count > 5:
            is_cheat = True
            reason = f"Gian lận (Phát hiện gian lận quá 5 lần)"
            break

        
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