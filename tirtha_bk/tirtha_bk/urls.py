"""tirtha_bk URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls import handler403, handler404, handler500
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path


PRE_URL = "project/tirtha/"

urlpatterns = [
    path("", include("tirtha.urls")),
    path(PRE_URL + "admin/", admin.site.urls),
] + static(PRE_URL + settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Only serve in testing. Otherwise use gunicorn or nginx
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Error handling
handler403 = "tirtha_bk.views.handler403"
handler404 = "tirtha_bk.views.handler404"
handler500 = "tirtha_bk.views.handler500"
