from django.urls import path, re_path
from django.conf import settings
from django.contrib import admin

from . import views


admin.site.site_url = "/" + settings.PRE_URL
admin.site.site_header = "Project Tirtha Admin"
admin.site.site_title = "Project Tirtha"
admin.site.index_title = "Admin Portal"

# NOTE: The `pre` variable is used to prefix the URLs with a common string.
pre = settings.PRE_URL

urlpatterns = [
    path(
        pre + "competition/", views.competition, name="competition"
    ),  # NOTE: Order matters
    path(pre + "howto/", views.howto, name="howto"),
    path(pre + "signin/", views.signin, name="signin"),
    path(pre + "verifyToken/", views.verifyToken, name="verifyToken"),  # Auth Redirect
    path(pre + "preUpload/", views.pre_upload_check, name="preUpload"),
    path(pre + "upload/", views.upload, name="upload"),
    path(pre + "search/", views.search, name="search"),
    # TODO: Disabling in favour of redirect for GSRuns
    # TODO: Refactor or remove
    # path(pre + "loadMesh/", views.loadMesh, name="loadMesh"),
    # path(pre + "loadRun/", views.loadRun, name="loadRun"),
    path(pre + "models/<str:vid>/", views.index, name="indexMesh"),
    path(pre + "models/<str:vid>/<str:runid>/", views.index, name="indexMesh"),
    re_path(
        rf"^{pre}(resolve/)?(?P<ark>ark:/?.*$)", views.resolveARK, name="resolveARK"
    ),  # LATE_EXP: Add support for `?info` and `??info` queries or something similar. Check ARK spec.
    path(f"{pre}", views.index, name="index"),
]
