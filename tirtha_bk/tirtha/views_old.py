# LATE_EXP: FIXME: This file requires a refactor. Too many indirections.
# To fix:
# * search()'s code is inconvenient (main.js).
# * See FIXME:s and LATE_EXP:s below.

import uuid
# import logging
from loguru import logger
from django.conf import settings
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

# OAuth Google SignIn
from authlib.integrations.django_client import OAuth
from google.auth.transport import requests
from google.oauth2 import id_token

# Local imports
from tirtha_bk.views import handler403, handler404

from .models import ARK, Contribution, Contributor, Image, Mesh, Run
from .tasks import post_save_contrib_imageops
from .utilsark import parse_ark


LOG_LOCATION = settings.LOG_LOCATION
PRE_URL = settings.PRE_URL
GOOGLE_LOGIN = settings.GOOGLE_LOGIN
ADMIN_MAIL = settings.ADMIN_MAIL
OAUTH_CONF = settings.OAUTH_CONF
BASE_URL = settings.BASE_URL
FALLBACK_ARK_RESOLVER = settings.FALLBACK_ARK_RESOLVER

# OAuth App Setup
oauth = OAuth()
google = oauth.register(
    "google",  # NOTE: More can be added
    client_id=OAUTH_CONF.get("OAUTH2_CLIENT_ID"),
    client_secret=OAUTH_CONF.get("OAUTH2_CLIENT_SECRET"),
    client_kwargs={
        "scope": OAUTH_CONF.get("OAUTH2_SCOPE"),
    },
    server_metadata_url=f"{OAUTH_CONF.get('OAUTH2_META_URL')}",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    redirect_uri=OAUTH_CONF.get("OAUTH2_REDIRECT_URI"),
)

logger.remove()
logger.add(
    LOG_LOCATION,
    level="NOTSET",  # or INFO, ERROR, etc.
    format="<green>{time:YYYY-MM-DD HH:mm:ss ZZ}</green> | <level>{level}: {message}</level>",
    colorize=True,  # Enable colorized output
    backtrace=True,  # Enable exception tracebacks
    diagnose=True,  # Enable variable values in tracebacks
    enqueue=True,   # Thread-safe logging
    rotation="100 MB",  # Rotate files when they reach 100 MB
    retention="45 days",  # Keep logs for 45 days
)

# Logger setup
# Logging
# logging.basicConfig(
#     level=logging.NOTSET,
#     format="%(asctime)s %(levelname)s %(message)s",
#     filename=LOG_LOCATION,
# )


def index(request, vid=None, runid=None):
    template = "tirtha/index.html"

    # Exclude hidden meshes (Includes default mesh)
    meshes = Mesh.objects.exclude(hidden=True).order_by("name")
    context = {
        "meshes": meshes,
        "signin_msg": "Please sign in to upload images.",
        "signin_class": "blur-form",
        "GOOGLE_CLIENT_ID": OAUTH_CONF.get("OAUTH2_CLIENT_ID"),
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
                .values_list("ark", "ended_at")
            )

            # Move the selected run to the front
            runs_arks = [
                (ark, ended_at) for ark, ended_at in runs_arks if ark != run.ark.ark
            ]
            runs_arks.insert(0, (run.ark.ark, run.ended_at))

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
                .values_list("ark", "ended_at")
            )

            # Move the selected run to the front
            runs_arks = [
                (ark, ended_at) for ark, ended_at in runs_arks if ark != run.ark.ark
            ]
            runs_arks.insert(0, (run.ark.ark, run.ended_at))

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
        logger.info("index -- Google login is disabled.")
        request.session["tirtha_user_info"] = None

    user_info = request.session.get("tirtha_user_info", None)
    logger.info(f"index -- User info: {user_info}")
    if user_info:
        logger.info("index -- User info exists. Continuing...")
        output, contrib = _signin(user_info)
        # Store user_info in session if contributor exists
        if contrib is not None:
            request.session["tirtha_user_info"] = user_info
            request.session.secure = True
            request.session.set_expiry(0)  # Session expires when browser closes
            context.update({"signin_msg": output})

            if not contrib.banned and contrib.active:
                context.update({"signin_class": ""})
    else:
        logger.info("index -- No user is signed in.")

    # Profile Image URL
    if user_info is not None:
        context["profile_image_url"] = user_info.get("picture")

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


def _signin(user_info: dict) -> tuple:
    """
    Retrieves or creates the contributor.

    """
    # For development only
    if not GOOGLE_LOGIN:
        logger.info("_signin -- Google login is disabled.")
        # Return default contributor
        contrib = Contributor.objects.get(email=ADMIN_MAIL)
        output = f"Signed-in as {ADMIN_MAIL}."
        return output, contrib

    logger.info(f"_signin -- Google login is enabled. Signing in user: {user_info}")
    # Get contributor info
    email = user_info.get("email")
    name = user_info.get("name")

    # NOTE: Treating email as unique ID, both for our DB and Google's
    # NOTE: Contributor is created as inactive | Manual activation required
    # CHECK: TODO: Allow auto-activation after testing
    # Get or create contributor
    contrib, _ = Contributor.objects.get_or_create(
        email=email, defaults={"active": False}
    )

    # If name has changed, update name
    if name != contrib.name:
        logger.info(f"Updating name for {email} from {contrib.name} to {name}.")
        contrib.name = name
        contrib.save()

    # Check if active
    output = f"Signed-in as {email}."
    if not contrib.active:
        logger.info(f"{email} is not active.")
        output = f"{email} is not active. Please contact the admin."

    # Check if banned
    if contrib.banned:
        logger.info(f"{email} has been banned.")
        output = f"{email} has been banned. Please contact the admin."

    return output, contrib


@require_GET
def signin(request):
    """
    First step in OAuth2.0 flow. Redirects to Google's OAuth2.0 consent screen.

    """
    # Build a full authorize callback URI using a new UUID
    new_state = str(uuid.uuid4())
    logger.info(f"signin -- Sent new_state: {new_state}")

    # Save the state and redirect to Google's OAuth2.0 consent screen
    request.session["auth_random_state"] = new_state
    redirect_uri = request.build_absolute_uri("/" + PRE_URL + "verifyToken/")

    return google.authorize_redirect(request, redirect_uri, state=new_state)


@require_GET
def verifyToken(request):
    """
    Second step in OAuth2.0 flow (Callback).
    Authorizes & sets auth token with relavent user information

    """
    request_state = request.session.get("auth_random_state", None)
    received_state = request.GET.get("state", None)

    key = f"_state_google_{received_state}"
    request.session[key] = {
        "data": {
            "state": received_state,
        }
    }

    logger.info(f"verifyToken -- Request state: {request_state}")
    logger.info(f"verifyToken -- Received state: {received_state}")

    if not received_state:
        logger.error("Received state is empty. Aborting sign-in.")
        return handler403(request)

    if request_state is None or received_state != request_state:
        logger.error("State mismatch. Aborting sign-in.")
        return handler403(request)

    try:
        logger.info("Verifying token...")
        # NOTE: dict_keys(['access_token', 'expires_in', 'scope', 'token_type', 'id_token', 'expires_at'])
        authlib_token = google.authorize_access_token(request)
        # logger.debug(f"verifyToken -- authlib_token: {authlib_token}")

        # Google OAuth2.0 token
        google_token = authlib_token["id_token"]
        # logger.debug(f"verifyToken -- google_token: {google_token}")

        # Verify id_token
        idinfo = id_token.verify_oauth2_token(
            google_token, requests.Request(), OAUTH_CONF.get("OAUTH2_CLIENT_ID")
        )
        logger.debug(f"verifyToken -- idinfo: {idinfo}")
        logger.info("Token verified. Adding to session...")
        request.session["tirtha_user_info"] = {
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }
        logger.info("Token added to session.")

    except ValueError as e:
        request.session["tirtha_user_info"] = None
        logger.error("ERROR in token verification:")
        logger.error(e)
        return handler403(request)

    return redirect(index)


@require_GET
def pre_upload_check(request) -> JsonResponse:
    """
    Pre-upload, checks if the mesh_vid is valid and if so, whether the mesh is "completed"

    """
    verbose_id = request.GET["mesh_vid"]

    # Authenticate contributor
    user_info = request.session.get("tirtha_user_info", None)
    if user_info is None:
        return JsonResponse(
            {"allowupload": False, "blur": True, "output": "Please sign in again."}
        )

    output, contrib = _signin(user_info)
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
def upload(request) -> JsonResponse:
    """
    Handles `Upload` form. Does the following:
    * Authenticates contributor
    * Checks if mesh exists
    * Creates new contribution & binds to mesh + contributor
    * Creates new images & binds to contribution

    """
    # Authenticate contributor
    user_info = request.session.get("tirtha_user_info", None)
    output, contrib = _signin(user_info)
    if contrib is None:
        return JsonResponse({"output": output})

    # Match mesh
    verbose_id = request.POST["mesh_vid"]
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
    print("Contribution created with ID:", contribution.ID)
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

    # Search by name, district, state, country
    search_query = (
        Q(name__icontains=query)
        | Q(country__icontains=query)
        | Q(state__icontains=query)
        | Q(district__icontains=query)
    )
    meshes = Mesh.objects.filter(search_query).exclude(hidden=True).order_by("name")

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
        logger.error(f"resolveARK -- ARK parsing error: {e}")
        return handler404(
            request, e
        )  # LATE_EXP: Maybe add a custom page saying the ARK is invalid

    try:
        # Try to find the ARK in the database
        ark = ARK.objects.get(ark=f"{naan}/{assigned_name}")
        return redirect("indexMesh", vid=ark.run.mesh.verbose_id, runid=ark.run.ID)
    except ARK.DoesNotExist:
        logger.error(f"resolveARK -- ARK not found: {ark}")
        return redirect(f"{FALLBACK_ARK_RESOLVER}/{ark}")


def competition(request):
    """
    Renders a static page with details about the Tirtha competition.

    """
    logger.info("competition -- Accessed.")
    if request.method == "GET":
        template = "tirtha/competition.html"
        return render(request, template)
    return handler403(request)  # FIXME: Change to 405


def howto(request):
    """
    Renders a static page with instructions on how to use Tirtha.

    """
    logger.info("howto -- Accessed.")
    if request.method == "GET":
        template = "tirtha/howto.html"
        return render(request, template)
    return handler403(request)  # FIXME: Change to 405
