from django.shortcuts import render, redirect 
from django.contrib.auth.decorators import login_required 
from django.http import Http404, HttpResponse 
from quiz.models.quiz import Exam, Question, Answer, Result, ResultDetail
from quiz.models.custom_user import CustomUser 


@login_required(login_url='/login/')
def exam_view(request, pk):
    try:
        exam = Exam.objects.get(pk=pk)
    except Exam.DoesNotExist:
        return Http404()
    
    if request.method == 'POST':
        print(request.POST)
        score = 0
        result = Result.objects.create(
            exam=exam,
            user=request.user,
            score=score
        )
        for q in exam.question_set.all():
            selected_answer_id = request.POST.get(f'question_{q.id}')
            if selected_answer_id and selected_answer_id != "":
                selected_answer = Answer.objects.get(id=selected_answer_id)
                result_detail = ResultDetail.objects.create(
                    result=result,
                    question=q,
                    answer=selected_answer,
                    is_correct=selected_answer.is_correct,
                )
                if selected_answer.is_correct:
                    score += 10 
            else:
                ResultDetail.objects.create(
                    result=result,
                    question=q,
                    answer=None,  
                    is_correct=False, 
                )

        result.score = score 
        result.save()
        return redirect('result', pk=result.id)


    return render(request, 'quiz/exam.html', {'exam': exam}, status=200)


@login_required(login_url='/login/')
def result_view(request, pk):
    try:
        result = Result.objects.get(pk=pk)
    except Result.DoesNotExist:
        return Http404()

    return render(request, 'quiz/result.html', {'result': result})