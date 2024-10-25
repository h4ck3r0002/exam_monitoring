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


