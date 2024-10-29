from django.shortcuts import render, redirect 
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from quiz.models.quiz import Exam, Monitor 
from django.http import Http404 


@csrf_exempt
def upload_video(request, pk):
    try:
        exam = Exam.objects.get(pk=pk)
    except Exam.DoesNotExist:
        return Http404() 

    if request.method == 'POST' and 'video_recording' in request.FILES:
        print(request.FILES)
        video_file = request.FILES['video_recording']
        # file_path = os.path.join('videos', 'recorded_video.webm')

        user = request.user 

        monitor = Monitor(
            user=user,
            exam=exam
        )
        monitor.video.save(f"{user.id}_{exam.id}_recorded_video.webm", video_file)
        monitor.save()
        
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'failed'}, status=400)


def monitor_list_view(request):
    exams = Exam.objects.order_by('-created_at')
    return render(request, 'quiz/monitor.html', {'exams': exams}, status=200)


def monitor_detail_view(request, pk):
    try:
        exam = Exam.objects.get(pk=pk)
    except Exam.DoesNotExist:
        return Http404()

    monitors = Monitor.objects.filter(exam=exam)
    return render(request, 'quiz/monitor-detail.html', {'monitors': monitors, 'exam': exam}, status=200)

