import cv2
import sys
import os
import django
import numpy as np
from django.conf import settings
from collections import deque 
import tensorflow as tf


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
django.setup()

from quiz.models.quiz import Monitor, Result

# Load the TensorFlow model
model = tf.keras.models.load_model("model.savedmodel") 
class_names = open("labels.txt", "r").readlines()

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

        # Resize and preprocess frame for the model
        image = cv2.resize(frame, (224, 224))
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1  # Normalize as per model's input requirement

        # Predict cheating status using the model
        prediction = model.predict(image)

        # # Assuming the model outputs a binary classification (0: no cheating, 1: cheating)
        # if prediction[0][0] > 0.5:  # Adjust threshold if necessary
        #     print("Cheating detected.")
        #     is_cheat = True
        #     reason = 'Detected cheating.'
        #     break

        # Kiểm tra cấu trúc đầu ra
        if isinstance(prediction, dict):
            output_key = list(prediction.keys())[0]  # Lấy tên của tensor đầu ra
            prediction = prediction[output_key]  # Lấy tensor từ từ điển

        # Tìm lớp dự đoán và điểm số độ tin cậy
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        if class_name == 'Cheat':
            is_cheat = True 
            reason = 'Detected cheating (Không có mặt trong màn hình, quay trái phải hoặc có nhiều người trong khung hình)'
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