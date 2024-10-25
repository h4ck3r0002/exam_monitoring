from celery import shared_task 
from quiz.models.quiz import * 
from .utils import detect_cheating 


@shared_task
def analyze_video_for_cheating(exam_id, video_path):
    exam = Exam.objects.get(pk=exam_id)
    cheating_detected = detect_cheating(video_path)
    if cheating_detected:
        print("Gian lận")
        pass 
        # lưu kết quả phát hiện gian lận 



