from django.urls import path 
from quiz.consumers import VideoStreamConsumer 


websocket_urlpatterns = [
    path('stream/', VideoStreamConsumer.as_asgi()),
]
