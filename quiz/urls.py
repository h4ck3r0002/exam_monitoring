from django.urls import path 
from quiz.views.index import * 
from quiz.views.quiz import * 


urlpatterns = [
    path('', index_view, name='index'),
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('exam/<pk>', exam_view, name='exam'),
    path('result/<pk>', result_view, name='result'),
]
