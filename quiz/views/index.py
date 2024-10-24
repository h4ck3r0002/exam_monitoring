from django.shortcuts import render, redirect 
from django.contrib.auth import login, authenticate, logout 
from django.contrib.auth.decorators import login_required 
from quiz.models.custom_user import CustomUser 
from django.contrib import messages 


def index_view(request):
    return render(request, 'quiz/index.html', status=200)


def register_view(request):
    if request.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        role = 'student'


    return render(request, 'quiz/register.html', status=200)


def login_view(request):
    if request.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
    
    return render(request, 'quiz/login.html', status=200)


@login_required(login_url='/login/')
def logout_view(request):
    logout(request)
    return redirect('login')


