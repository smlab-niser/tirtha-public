"""
Tirtha application views for handling mesh visualization, uploads, and authentication.

"""

from typing import Dict
import uuid
import logging

from django.conf import settings
from django.db.models import Q, Max
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from authlib.integrations.django_client import OAuth
from google.auth.transport import requests
from google.oauth2 import id_token

# Local imports
from tirtha_bk.views import handler403, handler404
from .models import ARK, Contribution, Contributor, Image, Mesh, Run
from .tasks import post_save_contrib_imageops
from .utilsark import parse_ark


# Configuration
LOG_LOCATION = settings.LOG_LOCATION
PRE_URL = settings.PRE_URL
GOOGLE_LOGIN = settings.GOOGLE_LOGIN
ADMIN_MAIL = settings.ADMIN_MAIL
OAUTH_CONF = settings.OAUTH_CONF
BASE_URL = settings.BASE_URL
FALLBACK_ARK_RESOLVER = settings.FALLBACK_ARK_RESOLVER

# OAuth setup
oauth = OAuth()
google = oauth.register(
    "google",
    client_id=OAUTH_CONF.get("OAUTH2_CLIENT_ID"),
    client_secret=OAUTH_CONF.get("OAUTH2_CLIENT_SECRET"),
    client_kwargs={"scope": OAUTH_CONF.get("OAUTH2_SCOPE")},
    server_metadata_url=f"{OAUTH_CONF.get('OAUTH2_META_URL')}",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    redirect_uri=OAUTH_CONF.get("OAUTH2_REDIRECT_URI"),
)

# Logging setup
logging.basicConfig(
    level=logging.NOTSET,
    format="%(asctime)s %(levelname)s %(message)s",
    filename=LOG_LOCATION,
)


def _authenticate_user(user_info):
    """
    Authenticates and retrieves or creates a contributor.

    Returns:
        tuple: (output_message, contributor_object)

    """
    if not GOOGLE_LOGIN:
        logging.info("Google login is disabled - using admin account")
        contrib = Contributor.objects.get(email=ADMIN_MAIL)
        return f"Signed-in as {ADMIN_MAIL}.", contrib

    if not user_info:
        return "Please sign in again.", None

    email = user_info.get("email")
    name = user_info.get("name")

    if not email:
        return "Invalid user information.", None

    # Get or create contributor (inactive by default)
    contrib, created = Contributor.objects.get_or_create(
        email=email, defaults={"active": False, "name": name}
    )

    # Update name if changed
    if name and name != contrib.name:
        logging.info(f"Updating name for {email} from {contrib.name} to {name}")
        contrib.name = name
        contrib.save()

    # Check status
    if contrib.banned:
        return f"{email} has been banned. Please contact the admin.", contrib

    if not contrib.active:
        return f"{email} is not active. Please contact the admin.", contrib

    return f"Signed-in as {email}.", contrib


def _get_mesh_context(mesh: Mesh, run: Run=None) -> Dict:
    """
    Helper to build mesh context for templates

    Args:
        mesh (Mesh): Mesh object
        run (Run, optional): Specific Run object. Defaults to None.

    Returns:
        dict: Context dictionary for rendering

    """
    context = {
        "mesh": mesh,
        "mesh_contribution_count": mesh.contributions.count(),
        "mesh_images_count": Image.objects.filter(contribution__mesh=mesh).count(),
        "orientation": f"{mesh.rotaZ}deg {mesh.rotaX}deg {mesh.rotaY}deg",
        "src": f"static/models/{mesh.ID}/published/{mesh.ID}__default.glb",
    }

    if run:
        runs_arks = list(
            mesh.runs.filter(status="Archived")
            .order_by("-ended_at")
            .values_list("ark", "ended_at")
        )
        # Move selected run to front
        runs_arks = [
            (ark, ended_at) for ark, ended_at in runs_arks if ark != run.ark.ark
        ]
        runs_arks.insert(0, (run.ark.ark, run.ended_at))

        context.update(
            {
                "orientation": f"{run.rotaZ}deg {run.rotaX}deg {run.rotaY}deg",
                "run": run,
                "run_contributor_count": run.contributors.count(),
                "run_images_count": run.images.count(),
                "run_ark_url": f"{BASE_URL}/{run.ark}",
                "runs_arks": runs_arks,
            }
        )
    else:
        context["run"] = None

    return context


def index(request, vid: str=None, runid: str=None):
    """
    Main view for displaying meshes and runs

    """
    template = "tirtha/index.html"

    # Get all non-hidden meshes with non-hidden latest run
    meshes = (
        Mesh.objects.exclude(hidden=True)
        .annotate(latest_run=Max("runs__ended_at", filter=Q(runs__hidden=False)))
        .order_by("latest_run")
    )

    # Base context
    context = {
        "meshes": meshes,
        "signin_msg": "Please sign in to upload images.",
        "signin_class": "blur-form",
        "GOOGLE_CLIENT_ID": OAUTH_CONF.get("OAUTH2_CLIENT_ID"),
    }

    # Handle specific run request
    if runid:
        try:
            run = Run.objects.get(ID=runid)

            if run.hidden or run.mesh.hidden:
                logging.warning(f"Attempt to access hidden Run or Mesh: {runid}")
                raise Run.DoesNotExist("Run or Mesh is hidden.")

            mesh_context = _get_mesh_context(run.mesh, run)
            context.update(mesh_context)
        except Run.DoesNotExist as e:
            return handler404(request, e)

    # Handle mesh request (with or without vid)
    else:
        try:
            if vid:
                mesh = Mesh.objects.get(verbose_id=vid)
            else:
                mesh = Mesh.objects.get(ID=settings.DEFAULT_MESH_ID)

            # Try to get latest archived run
            try:
                run = mesh.runs.filter(status="Archived").latest("ended_at")
                if run.hidden:
                    logging.warning(f"Attempt to access hidden Run: {run.ID}")
                    raise Run.DoesNotExist("Run is hidden.")
            except Run.DoesNotExist:
                run = None

            mesh_context = _get_mesh_context(mesh, run)
            context.update(mesh_context)

        except Mesh.DoesNotExist as e:
            return handler404(request, e)

    # Handle authentication
    user_info = request.session.get("tirtha_user_info")
    if user_info:
        output, contrib = _authenticate_user(user_info)
        if contrib:
            request.session["tirtha_user_info"] = user_info
            request.session.secure = True
            request.session.set_expiry(0)
            context["signin_msg"] = output
            context["profile_image_url"] = user_info.get("picture")

            if not contrib.banned and contrib.active:
                context["signin_class"] = ""

    return render(request, template, context)


@require_GET
def signin(request):
    """OAuth2.0 authorization redirect to Google."""
    new_state = str(uuid.uuid4())
    logging.info(f"OAuth signin initiated with state: {new_state}")

    request.session["auth_random_state"] = new_state
    redirect_uri = request.build_absolute_uri("/" + PRE_URL + "verifyToken/")

    return google.authorize_redirect(request, redirect_uri, state=new_state)


@require_GET
def verifyToken(request):
    """OAuth2.0 callback - verifies token and sets user session."""
    request_state = request.session.get("auth_random_state")
    received_state = request.GET.get("state")

    # Validate state for security
    if not received_state or request_state != received_state:
        logging.error("OAuth state mismatch - potential CSRF attack")
        return handler403(request)

    # Set up session key for authlib
    key = f"_state_google_{received_state}"
    request.session[key] = {"data": {"state": received_state}}

    try:
        # Get access token from Google
        authlib_token = google.authorize_access_token(request)
        google_token = authlib_token["id_token"]

        # Verify and decode the JWT token
        idinfo = id_token.verify_oauth2_token(
            google_token, requests.Request(), OAUTH_CONF.get("OAUTH2_CLIENT_ID")
        )

        # Store user info in session
        request.session["tirtha_user_info"] = {
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }
        logging.info(f"User authenticated: {idinfo.get('email')}")

    except ValueError as e:
        logging.error(f"Token verification failed: {e}")
        request.session["tirtha_user_info"] = None
        return handler403(request)

    return redirect(index)


@require_GET
def pre_upload_check(request):
    """Validates mesh and user before allowing upload."""
    verbose_id = request.GET.get("mesh_vid")
    if not verbose_id:
        return JsonResponse(
            {"allowupload": False, "blur": False, "output": "Mesh ID is required."}
        )

    # Authenticate user
    user_info = request.session.get("tirtha_user_info")
    if not user_info:
        return JsonResponse(
            {"allowupload": False, "blur": True, "output": "Please sign in again."}
        )

    output, contrib = _authenticate_user(user_info)
    if not contrib:
        return JsonResponse({"allowupload": False, "blur": True, "output": output})

    if contrib.banned:
        return JsonResponse(
            {
                "allowupload": False,
                "blur": True,
                "output": f"{contrib.email} has been banned. Please contact the admin.",
            }
        )

    if not contrib.active:
        return JsonResponse(
            {
                "allowupload": False,
                "blur": True,
                "output": f"{contrib.email} has not been activated. Please contact the admin.",
            }
        )

    # Validate mesh
    try:
        mesh = Mesh.objects.get(verbose_id__exact=verbose_id)
    except Mesh.DoesNotExist:
        return JsonResponse(
            {
                "allowupload": False,
                "blur": False,
                "output": "Specified model was not found in database.",
            }
        )

    if mesh.completed:
        return JsonResponse(
            {
                "allowupload": False,
                "blur": False,
                "output": "This model is not accepting contributions at the moment.",
            }
        )

    return JsonResponse({"allowupload": True, "output": "Mesh found!"})


@require_POST
def upload(request):
    """
    Handles image upload and creates contribution

    """
    # Authenticate user
    user_info = request.session.get("tirtha_user_info")
    output, contrib = _authenticate_user(user_info)
    if not contrib:
        return JsonResponse({"output": output})

    if contrib.banned or not contrib.active:
        return JsonResponse({"output": "Account not authorized for uploads."})

    # Get mesh
    verbose_id = request.POST.get("mesh_vid")
    if not verbose_id:
        return JsonResponse({"output": "Mesh ID is required."})

    try:
        mesh = Mesh.objects.get(verbose_id__exact=verbose_id)
    except Mesh.DoesNotExist:
        return JsonResponse({"output": "Mesh not found."})

    if mesh.completed:
        return JsonResponse({"output": "This mesh is not accepting contributions."})

    # Process images
    images = request.FILES.getlist("images")
    if not images:
        return JsonResponse({"output": "No images provided."})

    # Create contribution and images
    contribution = Contribution.objects.create(mesh=mesh, contributor=contrib)
    image_objs = [Image(image=image, contribution=contribution) for image in images]
    Image.objects.bulk_create(image_objs)

    # Update timestamps
    contribution.save()
    mesh.save()

    # Trigger background processing
    try:
        post_save_contrib_imageops.delay(str(contribution.ID))
    except Exception as e:
        logging.error(f"Failed to trigger image processing task: {e}")
        logging.error("Deleting contribution due to processing failure.")
        contribution.delete()

        return JsonResponse(
            {"status": "Error", "output": "Failed to process images. Please try again."}
        )

    return JsonResponse(
        {"status": "Success", "output": "Successfully uploaded. Thank you!"}
    )


@require_GET
def search(request):
    """Search meshes by name, country, state, or district."""
    query = request.GET.get("query", "").strip()

    if not query:
        return JsonResponse({"status": "No query provided", "meshes_json": {}})

    # Search across multiple fields
    search_query = (
        Q(name__icontains=query)
        | Q(country__icontains=query)
        | Q(state__icontains=query)
        | Q(district__icontains=query)
    )

    meshes = (
        Mesh.objects.filter(search_query)
        .exclude(hidden=True)
        .annotate(latest_run=Max("runs__ended_at"))
        .order_by("latest_run")
    )

    meshes_json = {}
    for mesh in meshes:
        meshes_json[mesh.name] = {
            "verbose_id": mesh.verbose_id,
            "thumb_url": mesh.thumbnail.url,
            "completed_msg": "Closed" if mesh.completed else "Open",
            "completed_col": "firebrick" if mesh.completed else "forestgreen",
        }

    status = "Mesh found!" if meshes else "Mesh not found!"
    return JsonResponse({"status": status, "meshes_json": meshes_json})


@require_GET
def resolveARK(request, ark: str):
    """
    Resolve ARK identifiers to mesh/run views.

    """
    try:
        naan, assigned_name = parse_ark(ark)
        ark_obj = ARK.objects.get(ark=f"{naan}/{assigned_name}")

        if ark_obj.run.hidden or ark_obj.run.mesh.hidden:
            logging.warning(f"Attempt to access hidden ARK: {ark}")
            raise ARK.DoesNotExist("ARK points to hidden content.")

        return redirect(
            "indexMesh", vid=ark_obj.run.mesh.verbose_id, runid=ark_obj.run.ID
        )
    except (ValueError, ARK.DoesNotExist) as e:
        logging.error(f"ARK resolution failed for {ark}: {e}")
        return redirect(f"{FALLBACK_ARK_RESOLVER}/{ark}")


@require_GET
def competition(request):
    """
    Competition information page

    """
    return render(request, "tirtha/competition.html")


@require_GET
def howto(request):
    """
    How-to instructions page

    """
    return render(request, "tirtha/howto.html")
