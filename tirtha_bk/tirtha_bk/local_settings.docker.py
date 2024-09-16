# NOTE: Rename to local_settings.py after editing
"""
Fields marked CHANGEME: need to be changed before deployment

"""
import os
from pathlib import Path
from django.core.management.utils import get_random_secret_key


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("SECRET_KEY")  # CHANGEME: NOTE: Keep this secret
SECRET_KEY = SECRET_KEY if SECRET_KEY else get_random_secret_key()

TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Kolkata")  # CHANGEME:

# SECURITY WARNING: do not run with debug turned on in production!
DEBUG = True  # NOTE: Set to False in production  # CHANGEME:
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin-allow-popups"
ALLOWED_HOSTS = ["localhost", "0.0.0.0", os.getenv("HOST_IP", "127.0.0.1")]  # CHANGEME:

# Application definition
INSTALLED_APPS = [
    "tirtha.apps.TirthaConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_cleanup.apps.CleanupConfig",  # For cleaning up orphaned files in media
    "django_extensions",
    "dbbackup",  # django-dbbackup
]

# Tirtha specific settings
BASE_DIR = Path(__file__).resolve().parent.parent
PRE_URL = os.getenv("PRE_URL", "")  # CHANGEME: e.g., "/tirtha/"

PROD_DIR = "/var/www/tirtha/prod/"  # Short term storage for current runs # CHANGEME:
LOG_DIR = f"{PROD_DIR}logs"
NFS_DIR = "/var/www/tirtha/archive/"  # Long term storage for old runs # CHANGEME: Does not need to use NFS and can be on the same system
ARCHIVE_ROOT = f"{NFS_DIR}archives"

# Static files
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]  # JS, CSS, images, favicon
STATIC_URL = PRE_URL + "static/"
STATIC_ROOT = os.path.join(PROD_DIR, "static")

# Media files
MEDIA_URL = PRE_URL + "media/"
MEDIA_ROOT = f"{PROD_DIR}media"
# STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Default attributes used to create default mesh & contributor
# NOTE: You will need to create the default contributor manually before running the server for the first time -> Needed in `post_migrate_create_defaults()` in `tirtha_bk/tirtha/signals.py.
DEFAULT_MESH_NAME = "NISER Meditation Center"  # Default mesh name to use while setting up | Will be shown on homepage
DEFAULT_MESH_ID = "9zpT9kVZwP9XxAbG"
ADMIN_NAME = os.getenv("DEFAULT_USER_NAME", "Tirtha Admin")  # CHANGEME:
ADMIN_MAIL = os.getenv("DEFAULT_USER_MAIL", "tadmin@example.com")  # CHANGEME:

# Sign in with Google
GOOGLE_LOGIN = os.getenv("GOOGLE_LOGIN", "False").lower() == "true"
GOOGLE_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID", ""
)  # CHANGEME: https://developers.google.com/identity/gsi/web/guides/overview
COOKIE_EXPIRE_TIME = 3600  # 1 hour
SESSION_COOKIE_SAMESITE = "Strict"
SESSION_COOKIE_SECURE = True if GOOGLE_LOGIN else False

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases
# NOTE: For help, see https://www.digitalocean.com/community/tutorials/how-to-set-up-django-with-postgres-nginx-and-gunicorn-on-ubuntu-22-04
DB_NAME = os.getenv("DB_NAME", "dbtirtha")  # CHANGEME:
DB_USER = os.getenv("DB_USER", "dbtirthauser")  # CHANGEME:
DB_PWD = os.getenv("DB_PWD", "docker")  # CHANGEME:
DB_HOST = os.getenv("DB_HOST", "db")  # CHANGEME:
DB_PORT = os.getenv("DB_PORT", "5432")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": DB_NAME,
        "USER": DB_USER,
        "PASSWORD": DB_PWD,
        "HOST": DB_HOST,
        "PORT": DB_PORT,
    }
}

# django-dbbackup
DBBACKUP_STORAGE = "django.core.files.storage.FileSystemStorage"
DBBACKUP_STORAGE_OPTIONS = {
    "location": f"{NFS_DIR}/db_backups/"
}  # CHANGEME: To store backups

## RabbitMQ + Celery
RMQ_USER = os.getenv("RMQ_USER", "rmqtirthauser")  # CHANGEME:
RMQ_PWD = os.getenv("RMQ_PWD", "rmqtirthapwd")  # CHANGEME:
RMQ_VHOST = os.getenv(
    "RMQ_VHOST", "rmqtirtha"
)  # CHANGEME: NOTE: You can also use the default vhost ("/").
CELERY_BROKER_URL = f"pyamqp://{RMQ_USER}:{RMQ_PWD}@localhost/{RMQ_VHOST}"
CELERY_TASK_ACKS_LATE = True  # To prevent tasks from being lost
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # Disable prefetching
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1  # NOTE: For memory release: https://stackoverflow.com/questions/17541452/celery-does-not-release-memory
CELERY_BROKER_CONNECTION_RETRY = False
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_CONNECTION_MAX_RETRIES = 10

## Worker-related settings
# GS
GS_MAX_ITER = 20_000
ALPHA_CULL_THRESH = 0.005  # Threshold to delete translucent gaussians - lower values remove more (usually better quality)
CULL_POST_DENS = False  # Disable culling after 15K steps

# MR
# NOTE: Defaulting to Meshroom 2021 for now. 2023 will require further changes
ALICEVISION_DIRPATH = BASE_DIR / "bin21"
NSFW_MODEL_DIRPATH = (
    BASE_DIR / "nn_models/nsfw_model/mobilenet_v2_140_224/"
)  # NOTE: See `Requirements` section in README.md
MANIQA_MODEL_FILEPATH = BASE_DIR / "nn_models/MANIQA/ckpt_kadid10k.pt"
OBJ2GLTF_PATH = "obj2gltf"  # NOTE: Ensure the binary is on system PATH
GLTFPACK_PATH = "gltfpack"  # NOTE: Ensure the binary is on system PATH
MESHOPS_MIN_IMAGES = 10  # CHANGEME: Minimum number of images required to run meshops
MESHOPS_CONTRIB_DELAY = 0.005  # 18 seconds for testing | Keep >= 1 hour(s) - CHANGEME: time to wait before running meshops after a new contribution
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10 MiB (each file max size - post compression)
DATA_UPLOAD_MAX_NUMBER_FILES = 1_000  # CHANGEME: Max number of files per upload

## ARK settings
BASE_URL = os.getenv(
    "BASE_URL",
    f"http://{os.getenv('HOST_IP', '0.0.0.0')}:{os.getenv('GUNICORN_PORT', '8000')}",
)  # CHANGEME: NOTE: No trailing "/" | e.g., http://127.0.0.1
ARK_NAAN = int(
    os.getenv("ARK_NAAN", "999999")
)  # CHANGEME: Integer | NOTE: NAAN - 999999 does not exist; CHECK: https://arks.org/about/testing-arks/
ARK_SHOULDER = os.getenv(
    "ARK_SHOULDER", "/a"
)  # CHANGEME: | CHECK: https://arks.org/about/testing-arks/
FALLBACK_ARK_RESOLVER = "https://n2t.net"
