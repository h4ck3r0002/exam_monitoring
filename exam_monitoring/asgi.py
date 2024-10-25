"""
ASGI config for exam_monitoring project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application 
from channels.routing import ProtocolTypeRouter, URLRouter 
from channels.auth import AuthMiddlewareStack 
from exam_monitoring.routing import websocket_urlpatterns 

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exam_monitoring.settings')

# application = get_asgi_application()
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    )
})
