from django.urls import path, re_path

from . import views

urlpatterns = [
    path("competition/", views.competition, name="competition"),  # NOTE: Order matters
    path("howto/", views.howto, name="howto"),
    path("googleAuth/", views.googleAuth, name="googleAuth"),
    path("preUpload/", views.pre_upload_check, name="preUpload"),
    path("upload/", views.upload, name="upload"),
    path("search/", views.search, name="search"),
    path("loadMesh/", views.loadMesh, name="loadMesh"),
    path("loadRun/", views.loadRun, name="loadRun"),
    path("models/<str:vid>/", views.index, name="indexMesh"),
    path("models/<str:vid>/<str:runid>/", views.index, name="indexMesh"),
    re_path(
        r"^(resolve/)?(?P<ark>ark:/?.*$)", views.resolveARK, name="resolveARK"
    ),  # LATE_EXP: Add support for `?info` and `??info` queries or something similar. Check ARK spec.
    path("", views.index, name="index"),
]
