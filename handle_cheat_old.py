import sys
import os
import django
import cv2
import numpy as np
from django.conf import settings
from keras.models import load_model

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
django.setup()

from quiz.models.quiz import Monitor, Result 


# Load model gian lận
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def process_video(monitor_id):
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

        # Tiền xử lý frame giống như mô hình dự đoán
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
        frame = (frame / 127.5) - 1

        # Dự đoán gian lận
        prediction = model.predict(frame)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Kiểm tra nếu mô hình phát hiện gian lận với xác suất cao
        if class_name == "cheat" and confidence_score > 0.9:  # Ví dụ: ngưỡng là 90%
            is_cheat = True
            break

    camera.release()

    # Cập nhật trạng thái gian lận trong cơ sở dữ liệu
    monitor.is_cheat = is_cheat
    monitor.save()

    result = Result.objects.get(exam=monitor.exam, user=monitor.user)
    result.is_cheat = is_cheat 
    result.is_done = True 
    result.save()

if __name__ == "__main__":
    monitor_id = int(sys.argv[1])
    process_video(monitor_id)
