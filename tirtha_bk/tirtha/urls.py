from django.urls import path, re_path
from django.conf import settings

from . import views


pre = settings.PRE_URL

urlpatterns = [
    path(pre + "competition/", views.competition, name="competition"),  # NOTE: Order matters
    path(pre + "howto/", views.howto, name="howto"),
    path(pre + "signin-google/", views.tokenAuth, name="tokenAuth"), #NOTE: for the authlib and token #TODO:CHANGE THE NAME OF THE FUNCTION as PER GOOGLE BACKEND CONFIGS
    path(pre + "signinGoogle/", views.google_login, name="signinGoogle"), #NOTE: for the authlib and token #TODO:CHANGE THE NAME OF THE FUNCTION
    path(pre + "logout/", views.logout, name="logout"), #NOTE: For the logout of user
    path(pre + "preUpload/", views.pre_upload_check, name="preUpload"),
    path(pre + "upload/", views.upload, name="upload"),
    path(pre + "search/", views.search, name="search"),
    path(pre + "loadMesh/", views.loadMesh, name="loadMesh"),
    path(pre + "loadRun/", views.loadRun, name="loadRun"),
    path(pre + "models/<str:vid>/", views.index, name="indexMesh"),
    path(pre + "models/<str:vid>/<str:runid>/", views.index, name="indexMesh"),
    re_path(
        rf"^{pre}(resolve/)?(?P<ark>ark:/?.*$)", views.resolveARK, name="resolveARK"
    ),  # LATE_EXP: Add support for `?info` and `??info` queries or something similar. Check ARK spec.
    path(f"{pre}", views.index, name="index"),
]
