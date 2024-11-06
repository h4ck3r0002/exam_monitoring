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


# Tải mô hình, scaler và label encoder
model = joblib.load('cheat_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
pca = joblib.load('pca.pkl')  # Tải PCA

def extract_hog_features(image):
    # Đảm bảo ảnh có kích thước 64x64
    image = cv2.resize(image, (64, 64))  # Điều chỉnh kích thước ảnh
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm='L2-Hys', visualize=True)
    return features

def predict_image(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hog_features = extract_hog_features(gray_image)

    hog_features = hog_features.reshape(1, -1)  # Reshape cho mô hình dự đoán
    hog_features = scaler.transform(hog_features)  # Chuẩn hóa dữ liệu
    hog_features = pca.transform(hog_features)  # Giảm chiều dữ liệu

    # Dự đoán
    prediction = model.predict(hog_features)
    probabilities = model.decision_function(hog_features)  # Lấy giá trị quyết định
    probabilities = 1 / (1 + np.exp(-probabilities))  # Chuyển đổi thành xác suất

    # Đảm bảo chỉ có "cheat" hoặc "normal"
    label = label_encoder.inverse_transform(prediction)[0]
    if label not in ["cheat", "normal"]:
        label = "unknown"  # Trường hợp bất ngờ, gán là "unknown"

    return label, probabilities[0]


def process_video(monitor_id):
    reason = ''

    # Lấy đối tượng Monitor
    monitor = Monitor.objects.get(id=monitor_id)
    video_path = monitor.video.path

    # Mở video
    camera = cv2.VideoCapture(video_path)
    is_cheat = False

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        # Dự đoán với từng khung hình
        label, prob = predict_image(frame)
        print(label, prob)

        # Nếu phát hiện gian lận
        if label == "cheat" and float(prob) > 0.8:
            is_cheat = True
            reason = f"Phát hiện gian lận (Lý do: Không có người xuất hiện trong khung hình hoặc có nhiều hơn 1 nguời trong khung hình)"
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
    process_video(monitor_id)