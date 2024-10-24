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

    # print(exam.questions.all())

    questions = Question.objects.filter(exam=exam)
    
    # # AI Generation 
    # if request.method == 'POST':
    #     score = 0
    #     for question in exam.questions.all():
    #         selected_answer_id = request.POST.get(f'question_{question.id}')
    #         if selected_answer_id:
    #             selected_answer = Answer.objects.get(id=selected_answer_id)
    #             result_detail = ResultDetail(
    #                 result=None,  # Tạm thời, sẽ cập nhật sau
    #                 question=question,
    #                 answer=selected_answer,
    #                 is_correct=selected_answer.is_correct
    #             )
    #             if selected_answer.is_correct:
    #                 score += 1
    #     result = Result.objects.create(
    #         user=request.user,
    #         exam=exam,
    #         score=score
    #     )
    #     for detail in ResultDetail.objects.filter(result=None):
    #         detail.result = result
    #         detail.save()
    #     return redirect('result', result_id=result.id)
    # # AI Generation


    return render(request, 'quiz/exam.html', {'exam': exam, 'questions': questions}, status=200)


@login_required(login_url='/login/')
def result_view(request, pk):
    try:
        result = Result.objects.get(pk=pk)
    except Result.DoesNotExist:
        return Http404()

    return render(request, 'quiz/result.html', {'result': result})