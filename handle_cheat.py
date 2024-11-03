# import cv2
# import sys
# import os
# import django
# import numpy as np
# from django.conf import settings
# from collections import deque

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
# django.setup()

# from quiz.models.quiz import Monitor, Result

# # Cài đặt bộ phát hiện khuôn mặt của OpenCV
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def process_video(monitor_id):
#     reason = ''
    
#     # Lấy đối tượng Monitor
#     monitor = Monitor.objects.get(id=monitor_id)
#     video_path = monitor.video.path

#     # Mở video
#     camera = cv2.VideoCapture(video_path)
#     is_cheat = False
#     face_turn_count = 0
#     max_face_turns = 3  # Ngưỡng quay đầu
#     head_positions = deque(maxlen=5)  # Lưu các vị trí đầu gần nhất

#     while camera.isOpened():
#         ret, frame = camera.read()
#         if not ret:
#             break

#         # Chuyển ảnh sang xám để phát hiện khuôn mặt
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Kiểm tra nếu không có người trong camera
#         if len(faces) == 0:
#             print("Không có người xuất hiện trong camera.")
#             reason = 'Không có người xuất hiện trong camera.'
#             is_cheat = True
#             break
#         elif len(faces) > 1:
#             print("Phát hiện nhiều hơn 1 người trong camera.")
#             is_cheat = True
#             reason = 'Phát hiện nhiều hơn 1 người trong camera.'
#             break
#         else:
#             # Theo dõi vị trí đầu
#             for (x, y, w, h) in faces:
#                 face_center = (x + w // 2, y + h // 2)
#                 head_positions.append(face_center)

#                 # Kiểm tra quay đầu trái/phải
#                 if len(head_positions) == 5:
#                     left_right_movements = [head_positions[i][0] - head_positions[i - 1][0] for i in range(1, 5)]
#                     if all(movement > 0 for movement in left_right_movements) or all(movement < 0 for movement in left_right_movements):
#                         face_turn_count += 1
#                         head_positions.clear()

#                     if face_turn_count > max_face_turns:
#                         print("Phát hiện quay đầu nhiều lần.")
#                         is_cheat = True
#                         reason = 'Phát hiện quay đầu nhiều lần. (trên 3)'
#                         break

#     camera.release()

#     # Cập nhật trạng thái gian lận trong cơ sở dữ liệu
#     monitor.is_cheat = is_cheat
#     monitor.reason = reason
#     monitor.save()

#     print(monitor.exam)
#     print(monitor.user)

#     result = Result.objects.get(exam=monitor.exam, user=monitor.user)
#     result.is_cheat = is_cheat
#     result.reason = reason
#     result.is_done = True
#     result.save()

# if __name__ == "__main__":
#     monitor_id = int(sys.argv[1])
#     process_video(monitor_id)




import cv2
import sys
import os
import django
import numpy as np
from django.conf import settings
from collections import deque

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')
django.setup()

from quiz.models.quiz import Monitor, Result

# Cài đặt bộ phát hiện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_video(monitor_id):
    reason = ''
    
    # Lấy đối tượng Monitor
    monitor = Monitor.objects.get(id=monitor_id)
    video_path = monitor.video.path

    # Mở video
    camera = cv2.VideoCapture(video_path)
    is_cheat = False
    face_turn_count = 0
    max_face_turns = 3  # Ngưỡng quay đầu
    head_positions = deque(maxlen=5)  # Lưu các vị trí đầu gần nhất
    no_face_frames = 0  # Đếm số khung hình không phát hiện khuôn mặt
    max_no_face_frames = 10  # Số khung hình liên tiếp không có khuôn mặt trước khi đưa ra cảnh báo

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        # Chuyển ảnh sang xám để phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Kiểm tra nếu không có người trong camera
        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames > max_no_face_frames:
                print("Không có người xuất hiện trong camera.")
                reason = 'Không có người xuất hiện trong camera.'
                is_cheat = True
                break
        elif len(faces) > 1:
            print("Phát hiện nhiều hơn 1 người trong camera.")
            is_cheat = True
            reason = 'Phát hiện nhiều hơn 1 người trong camera.'
            break
        # else:
        #     # Reset no_face_frames vì đã phát hiện thấy khuôn mặt
        #     no_face_frames = 0

        #     # Theo dõi vị trí đầu
        #     for (x, y, w, h) in faces:
        #         face_center = (x + w // 2, y + h // 2)
        #         head_positions.append(face_center)

        #         # Kiểm tra quay đầu trái/phải
        #         if len(head_positions) == 5:
        #             left_right_movements = [head_positions[i][0] - head_positions[i - 1][0] for i in range(1, 5)]
        #             if all(movement > 0 for movement in left_right_movements) or all(movement < 0 for movement in left_right_movements):
        #                 face_turn_count += 1
        #                 head_positions.clear()

        #             if face_turn_count > max_face_turns:
        #                 print("Phát hiện quay đầu nhiều lần.")
        #                 is_cheat = True
        #                 reason = 'Phát hiện quay đầu nhiều lần. (trên 3)'
        #                 break

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
