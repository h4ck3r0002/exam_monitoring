from django.urls import path 
from .consumers import VideoStreamConsumer 


websocket_urlpatterns = [
    path('stream/', VideoStreamConsumer.as_asgi()),
]
