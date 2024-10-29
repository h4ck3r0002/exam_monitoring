from django.http import StreamingHttpResponse 
import cv2 
from django.shortcuts import render, redirect 


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    
    def __del__(self):
        self.video.release() 


    def get_frame(self):
        ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
    

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
        content_type='multipart/x-mixed-replace; boundary=frame')


def monitor_view(request):
    return render(request, 'quiz/monitor.html', status=200)





# views.py
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_frame(request):
    if request.method == 'POST':
        # Lấy frame từ request và giải mã từ base64
        data = request.body.decode('utf-8')
        frame_data = data.split(",")[1]
        frame = base64.b64decode(frame_data)

        # Xử lý hoặc lưu frame tại đây
        # Ví dụ: Ghi frame vào file hoặc gửi tiếp đến trang giám sát 
        # Lưu frame vào file (lưu ý đường dẫn tới thư mục lưu frame)
        file_path = 'latest_frame.jpg'
        with default_storage.open(file_path, 'wb') as f:
            f.write(frame)

        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'failed'}, status=400)


import os
from django.http import JsonResponse
from django.core.files.storage import default_storage

def latest_frame(request):
    # Đọc frame từ file hoặc từ bộ nhớ tạm
    try:
        with default_storage.open('latest_frame.jpg', 'rb') as f:
            frame_data = base64.b64encode(f.read()).decode('utf-8')
        return JsonResponse({'frame': frame_data})
    except FileNotFoundError:
        return JsonResponse({'frame': ''})

