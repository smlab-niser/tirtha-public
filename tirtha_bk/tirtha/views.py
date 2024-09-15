# LATE_EXP: FIXME: This file requires a refactor. Too many indirections.
# To fix:
# * Too many different contexts for signin() and googleAuth().
# * search()'s code is inconvenient (main.js).
# * See FIXME:s and LATE_EXP:s below.
from urllib.parse import unquote

import pytz
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST


# Local imports
from tirtha_bk.views import handler403, handler404

from .models import ARK, Contribution, Contributor, Image, Mesh, Run
from .tasks import post_save_contrib_imageops
from .utilsark import parse_ark

from authlib.integrations.django_client import OAuth # for the oauth setup
import uuid

PRE_URL = settings.PRE_URL
GOOGLE_LOGIN = settings.GOOGLE_LOGIN
ADMIN_MAIL = settings.ADMIN_MAIL
appConf = settings.APP_CONF
BASE_URL = settings.BASE_URL
FALLBACK_ARK_RESOLVER = settings.FALLBACK_ARK_RESOLVER

# oauth app setup
oauth = OAuth()
oauth.register(
    "google",
    client_id=appConf.get("OAUTH2_CLIENT_ID"),
    client_secret=appConf.get("OAUTH2_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'{appConf.get("OAUTH2_META_URL")}',
)

def index(request, vid=None, runid=None):
    template = "tirtha/index.html"

    # Exclude hidden meshes (Includes default mesh)
    meshes = Mesh.objects.exclude(hidden=True).order_by("name")
    context = {
        "meshes": meshes,
        "signin_msg": "Please sign in to upload images.",
        "signin_class": "blur-form",
        "GOOGLE_CLIENT_ID": appConf.get("OAUTH2_CLIENT_ID"),
    }

    """
    NOTE: Ideally, only `runid` is required. But to allow a default mesh per site
    (e.g., to start with, or to test, or to populate website with default meshes etc.),
    we also allow `meshid` to be passed in the URL. If both are passed, `runid` is
    used.

    """
    if runid is not None:        
        try:
            run = Run.objects.get(ID=runid)
            runs_arks = list(
                run.mesh.runs.filter(status="Archived")
                .order_by("-ended_at")
                .values_list("ark", flat=True)
            )

            # Move the selected run to the front
            if run.ark.ark in runs_arks:
                runs_arks.remove(run.ark.ark)
                runs_arks.insert(0, run.ark.ark)

            context.update(
                {
                    "orientation": f"{run.rotaZ}deg {run.rotaX}deg {run.rotaY}deg",
                    "run": run,
                    "run_contributor_count": int(run.contributors.count()),
                    "run_images_count": int(run.images.count()),
                    "run_ark_url": f"{BASE_URL}/{run.ark}",
                    "runs_arks": runs_arks,
                }
            )
        except Run.DoesNotExist as e:
            return handler404(request, e)

    elif runid is None:
        if vid is None:
            mesh = Mesh.objects.get(ID=settings.DEFAULT_MESH_ID)
        else:
            try:
                mesh = Mesh.objects.get(verbose_id=vid)
            except Mesh.DoesNotExist as e:
                return handler404(request, e)

        # Add mesh info
        context.update(
            {
                "mesh": mesh,
                "mesh_contribution_count": mesh.contributions.count(),
                "mesh_images_count": Image.objects.filter(
                    contribution__mesh=mesh
                ).count(),
                "orientation": f"{mesh.rotaZ}deg {mesh.rotaX}deg {mesh.rotaY}deg",
                "src": f"static/models/{mesh.ID}/published/{mesh.ID}__default.glb",
            }
        )

        # Check and add run info
        try:
            # Check if a run exists for the mesh
            run = mesh.runs.filter(status="Archived").latest("ended_at")
            runs_arks = list(
                mesh.runs.filter(status="Archived")
                .order_by("-ended_at")
                .values_list("ark", flat=True)
            )

            # Move the selected run to the front
            if run.ark.ark in runs_arks:
                runs_arks.remove(run.ark.ark)
                runs_arks.insert(0, run.ark.ark)

        except Run.DoesNotExist:
            run = None

        if run:
            context.update(
                {
                    "orientation": f"{run.rotaZ}deg {run.rotaX}deg {run.rotaY}deg",
                    "run": run,
                    "run_contributor_count": int(run.contributors.count()),
                    "run_images_count": int(run.images.count()),
                    "run_ark_url": f"{BASE_URL}/{run.ark}",
                    "runs_arks": runs_arks,
                }
            )
        else:
            context.update(
                {
                    "run": None,
                }
            )

    # Check if contributor is signed in
    # For development only
    if not GOOGLE_LOGIN:
        token = "INSECURE-DEFAULT-TOKEN"
        request.session["auth_token"] = token

    token = request.session.get("auth_token", None)
    if token:
        output, contrib = _signin(token)
        # Store JWT in session
        if contrib is not None:
            request.session["auth_token"] = token
            request.session.secure = True
            request.session.set_expiry(0)  # Session expires when browser closes
            context.update({"signin_msg": output})

            if not contrib.banned and contrib.active:
                context.update({"signin_class": ""})

    # block for profile image url
    if token is not None:
        context["profile_image_url"] = token.get("userinfo").get(
            "picture"
        )  # adding the currently signed in profile image to the context

    return render(request, template, context)


# TODO: FIXME: Commented out since no XHR to accommodate GS runs
# @require_GET
# def loadMesh(request):
#     """
#     Allows AJAX requests to load mesh.

#     """
#     vid = request.GET.get("vid", None)

#     try:
#         mesh = Mesh.objects.get(verbose_id=vid)
#         runs_arks = list(
#             mesh.runs.filter(status="Archived")
#             .order_by("-ended_at")
#             .values_list("ark", flat=True)
#         )
#         # Get latest successful run for mesh (among Run.status == "Archived")
#         try:
#             run = mesh.runs.filter(status="Archived").latest("ended_at")
#         except Run.DoesNotExist:
#             run = None

#         # FIXME: LATE_EXP: Maybe remove default meshes and only allow runs to be loaded.
#         # Having both is counter-intuitive.
#         data = {
#             "status": "Mesh found!",
#             "mesh": {
#                 "status": mesh.status,
#                 "has_run": True if run else False,
#                 "src": run.ark.url
#                 if run
#                 else PRE_URL + f"static/models/{mesh.ID}/published/{mesh.ID}__default.glb",
#                 "prev_url": mesh.preview.url,
#                 "name": mesh.name,
#                 "desc": mesh.description,
#                 "last_recons": str(
#                     mesh.reconstructed_at.astimezone(
#                         pytz.timezone("Asia/Kolkata")
#                     ).strftime("%B %d, %Y")
#                 )
#                 if mesh.reconstructed_at
#                 else "Not reconstructed yet.",
#                 "contrib_type": "run" if run else "mesh",
#                 "runs_arks": runs_arks if runs_arks else ["N.A."],
#                 "run_ark": f"{run.ark}" if run else "N.A.",
#                 "run_ark_url": f"{BASE_URL}/{run.ark}" if run else "javascript:;",
#                 "contrib_count": int(run.contributors.count())
#                 if run
#                 else mesh.contributions.count(),
#                 "images_count": int(run.images.count())
#                 if run
#                 else Image.objects.filter(contribution__mesh=mesh).count(),
#                 "orientation": f"{run.rotaZ}deg {run.rotaX}deg {run.rotaY}deg"
#                 if run
#                 else f"{mesh.rotaZ}deg {mesh.rotaX}deg {mesh.rotaY}deg",
#             },
#         }
#     except Mesh.DoesNotExist:
#         data = {"status": "Mesh not found!", "mesh": None}

#     return JsonResponse(data)


# @require_GET
# def loadRun(request):
#     """
#     Allows AJAX requests to load run.

#     """
#     runark = request.GET.get("runark", None)
#     runark = "ark:/" + unquote(runark)

#     try:
#         naan, assigned_name = parse_ark(runark)
#         ark = ARK.objects.get(ark=f"{naan}/{assigned_name}")
#         run = ark.run
#         data = {
#             "status": "Run found!",
#             "run": {
#                 "mesh_src": run.ark.url,
#                 "orientation": f"{run.rotaZ}deg {run.rotaX}deg {run.rotaY}deg",
#                 "ended_at": str(
#                     run.ended_at.astimezone(pytz.timezone("Asia/Kolkata")).strftime(
#                         "%B %d, %Y"
#                     )
#                 ),
#                 "contrib_count": int(run.contributors.count()),
#                 "images_count": int(run.images.count()),
#                 "contrib_type": "run",
#                 "run_ark": f"{run.ark}",
#                 "run_ark_url": f"{BASE_URL}/{run.ark}",
#                 "mesh_name": run.mesh.name,
#                 "runid": run.ID,
#             },
#         }

#     except Run.DoesNotExist as e:
#         data = {"status": "Run not found!", "run": None}

#     return JsonResponse(data)


def _signin(token, create=False):
    """
    Handles token-based authentication.
    Verifies the token and retrieves or creates the contributor.
    """
    # For development only
    if not GOOGLE_LOGIN:
        # Return default contributor
        contrib = Contributor.objects.get(email=ADMIN_MAIL)
        output = f"Signed-in as {ADMIN_MAIL}."
        return output, contrib
    
    try:
        contrib = None

        # Get contributor info
        payload = token.get("userinfo")
        email = payload.get("email")
        name = payload.get("name")
        # NOTE: Treating email as unique ID, both for our DB and Google's
        # NOTE: Contributor is created as inactive | Manual activation required
        # CHECK: TODO: Allow auto-activation after testing

        # Get or create contributor
        contributor, created = Contributor.objects.get_or_create(
            email=email, name=name, defaults={"active": False}
        )
        
        # If name has changed, update name
        if name != contrib.name:
            contrib.name = name
            contrib.save()

        # Check if active
        output = f"Signed-in as {email}."
        if not contrib.active:
            output = f"{email} is not active. Please contact the admin."

        # Check if banned
        if contrib.banned:
            output = f"{email} has been banned. Please contact the admin."

    except ValueError:  # Invalid token
        output = "Please sign in again."

    return output, contrib


@require_GET
def googleAuth(request):
    """
    Sets auth token cookie post Google Auth.

    """
    token = request.GET["token"]
    output, contrib = _signin(token, create=True)

    context = {"output": output, "banned": False, "blur": False}

    # Store JWT in session
    if contrib is not None:
        request.session["auth_token"] = token
        request.session.set_expiry(0)  # Session expires when browser closes

        if contrib.banned or not contrib.active:
            context.update({"blur": True})

    return JsonResponse(context)


@require_GET
def pre_upload_check(request):
    """
    Pre-upload, checks if the mesh_vid is valid and if so, whether the mesh is "completed"

    """
    verbose_id = request.GET["mesh_vid"]

    # Authenticate contributor
    token = request.session.get("auth_token", None)
    if token is None:
        return JsonResponse(
            {"allowupload": False, "blur": True, "output": "Please sign in again."}
        )

    output, contrib = _signin(token)
    if contrib is None:
        return JsonResponse({"allowupload": False, "blur": True, "output": output})

    if contrib.banned:
        output = f"{contrib.email} has been banned. Please contact the admin."
        return JsonResponse({"allowupload": False, "blur": True, "output": output})

    if not contrib.active:
        output = f"{contrib.email} has not been activated. Please contact the admin."
        return JsonResponse({"allowupload": False, "blur": True, "output": output})

    # Check if mesh exists
    try:
        mesh = Mesh.objects.get(verbose_id__exact=verbose_id)
    except Mesh.DoesNotExist:
        output = "Specified model was not found in database."
        return JsonResponse({"allowupload": False, "blur": False, "output": output})

    # Check if mesh is accepting contributions
    if mesh.completed:
        output = "This model is not accepting contributions at the moment."
        return JsonResponse({"allowupload": False, "blur": False, "output": output})

    return JsonResponse({"allowupload": True, "output": "Mesh found!"})


@require_POST
def upload(request):
    """
    Handles `Upload` form. Does the following:
    * Authenticates contributor
    * Checks if mesh exists
    * Creates new contribution & binds to mesh + contributor
    * Creates new images & binds to contribution

    """
    # Authenticate contributor
    token = request.session.get("auth_token", None)
    output, contrib = _signin(token)
    if contrib is None:
        return JsonResponse({"output": output})

    # Match mesh
    verbose_id = request.POST["mesh_vid"]

    # NOTE: try-block not needed due to pre_upload_check
    mesh = Mesh.objects.get(verbose_id__exact=verbose_id)

    # Create Contribution
    contribution = Contribution.objects.create(mesh=mesh, contributor=contrib)

    # Create Images & attach to Contribution
    images = request.FILES.getlist("images")
    image_objs = [Image(image=image, contribution=contribution) for image in images]
    # NOTE: bulk_create() is faster than creating one-by-one and does not trigger signals
    # LATE_EXP: Test abulk_create() (async) for performance improvements
    Image.objects.bulk_create(image_objs)
    contribution.save()
    mesh.save()  # Updates mesh.updated_at
    post_save_contrib_imageops.delay(
        str(contribution.ID)
    )  # Send signal to trigger ImageOps

    output = "Successfully uploaded. Thank you!"
    return JsonResponse({"status": "Success", "output": output})


@require_GET
def search(request):
    # LATE_EXP: Marked for refactor + main.js
    data = {"status": "Mesh not found!", "meshes_json": None}

    query = request.GET.get("query", None)
    meshes = (
        Mesh.objects.filter(name__icontains=query).exclude(hidden=True).order_by("name")
    )
    meshes_json = dict()

    for mesh in meshes:
        meshes_json[mesh.name] = {
            "verbose_id": mesh.verbose_id,
            "thumb_url": mesh.thumbnail.url,
            "completed_msg": "Closed" if mesh.completed else "Open",
            "completed_col": "firebrick" if mesh.completed else "forestgreen",
        }

    if meshes:
        data = {"status": "Mesh found!", "meshes_json": meshes_json}

    return JsonResponse(data)


@require_GET
def resolveARK(request, ark: str):
    """
    NOTE: Adapted from arklet

    """
    # LATE_EXP: Add support for `?info` and `??info`
    try:
        naan, assigned_name = parse_ark(ark)
    except ValueError as e:
        return handler404(
            request, e
        )  # LATE_EXP: Maybe add a custom page saying the ARK is invalid

    try:
        # Try to find the ARK in the database
        ark = ARK.objects.get(ark=f"{naan}/{assigned_name}")
        return redirect("indexMesh", vid=ark.run.mesh.verbose_id, runid=ark.run.ID)
    
    except ARK.DoesNotExist as e:
        return redirect(f"{FALLBACK_ARK_RESOLVER}/{ark}")


def competition(request):
    """
    Renders a static page with details about the Tirtha competition.

    """
    if request.method == "GET":
        template = "tirtha/competition.html"
        return render(request, template)
    return handler403(request)  # FIXME: Change to 405


def howto(request):
    """
    Renders a static page with instructions on how to use Tirtha.

    """
    if request.method == "GET":
        template = "tirtha/howto.html"
        return render(request, template)
    return handler403(request)  # FIXME: Change to 405
