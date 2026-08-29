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
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin-allow-popups"
_allowed = os.getenv("ALLOWED_HOSTS")
if _allowed:
    ALLOWED_HOSTS = [h.strip() for h in _allowed.split(",") if h.strip()]
else:
    ALLOWED_HOSTS = [
        "localhost",
        "0.0.0.0",
        os.getenv("HOST_IP", "127.0.0.1"),
    ]  # CHANGEME:

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
MIDDLEWARE.append("tirtha.middleware.SecurityHeadersMiddleware")

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
PRE_URL = os.getenv("PRE_URL", "project/tirtha/")  # CHANGEME: e.g., "/tirtha/"

PROD_DIR = os.getenv(  # Short term storage for current runs # CHANGEME:
    "PROD_DIR", "/var/www/tirtha/prod/"
)
NFS_DIR = os.getenv(  # Long term storage for old runs # CHANGEME: Does not need to use NFS and can be on the same system
    "NFS_DIR", "/var/www/tirtha/archive/"
)
ARCHIVE_ROOT = f"{NFS_DIR}archives"
LOG_DIR = f"{PROD_DIR}logs"
LOG_LOCATION = LOG_DIR + "/django.log"
ADMIN_LOG_LOCATION = LOG_DIR + "/admin.log"

# Static files
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]  # JS, CSS, images, favicon
STATIC_URL = os.getenv("STATIC_URL", "/" + PRE_URL + "/static/")
STATIC_ROOT = os.path.join(os.getenv("PROD_DIR", PROD_DIR), "static")

# Media files
MEDIA_URL = os.getenv("MEDIA_URL", PRE_URL + "media/")
MEDIA_ROOT = os.path.join(os.getenv("PROD_DIR", PROD_DIR), "media")
# STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Default attributes used to create default mesh & contributor
# NOTE: You will need to create the default contributor manually before running the server for the first time -> Needed in `post_migrate_create_defaults()` in `tirtha_bk/tirtha/signals.py.
DEFAULT_MESH_NAME = os.getenv("DEFAULT_MESH_NAME", "Meditation Center")
DEFAULT_MESH_ID = os.getenv("DEFAULT_MESH_ID", "9zpT9kVZwP9XxAbG")
ADMIN_NAME = os.getenv("DEFAULT_USER_NAME", "Tirtha Admin")  # CHANGEME:
ADMIN_MAIL = os.getenv("DEFAULT_USER_MAIL", "tadmin@example.com")  # CHANGEME:

# Sign in with Google
GOOGLE_LOGIN = os.getenv("GOOGLE_LOGIN", "False").lower() == "true"
GOOGLE_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID", ""
)  # CHANGEME: https://developers.google.com/identity/gsi/web/guides/overview
GOOGLE_CLIENT_SECRET = os.getenv(
    "GOOGLE_CLIENT_SECRET", ""
)  # CHANGEME: https://developers.google.com/identity/gsi/web/guides/overview
COOKIE_EXPIRE_TIME = int(os.getenv("COOKIE_EXPIRE_TIME", "3600"))  # 1 hour
SESSION_COOKIE_SAMESITE = os.getenv(
    "SESSION_COOKIE_SAMESITE", "Lax"
)  # NOTE: Lax needed for authlib | See: https://docs.djangoproject.com/en/5.1/ref/settings/#std-setting-SESSION_COOKIE_SAMESITE
SESSION_COOKIE_SECURE = True if GOOGLE_LOGIN else False
CSRF_COOKIE_SAMESITE = "Strict"
CSRF_COOKIE_SECURE = True
SECURE_SSL_REDIRECT = True
SECURE_HSTS_PRELOAD = True
SECURE_HSTS_SECONDS = (
    31536000  # 1 year # NOTE: If sometime HTTPS is disabled, this should be removed
)
SECURE_BROWSER_XSS_FILTER = True
SECURE_HSTS_INCLUDE_SUBDOMAINS = True

# OAuth app config
OAUTH_CONF = {
    "OAUTH2_CLIENT_ID": GOOGLE_CLIENT_ID,
    "OAUTH2_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
    "OAUTH2_META_URL": "https://accounts.google.com/.well-known/openid-configuration",
    "OAUTH2_SCOPE": "openid profile email",
    "OAUTH2_REDIRECT_URI": "",  # CHANGEME: e.g., "https://yourdomain.com/project/tirtha/verifyToken/"
}

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases
# NOTE: For help, see https://www.digitalocean.com/community/tutorials/how-to-set-up-django-with-postgres-nginx-and-gunicorn-on-ubuntu-22-04
DB_NAME = os.getenv("DB_NAME", "dbtirtha")  # CHANGEME:
DB_USER = os.getenv("DB_USER", "dbtirthauser")  # CHANGEME:
DB_PWD = os.getenv("DB_PWD", "")  # CHANGEME: set DB password
DB_HOST = os.getenv("DB_HOST", "localhost")  # CHANGEME:
DB_PORT = os.getenv("DB_PORT", "")

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

# django-dbbackup (migrated to Django STORAGES)
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": MEDIA_ROOT},
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
        "OPTIONS": {"location": STATIC_ROOT},
    },
    "dbbackup": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {
            "location": os.getenv("DBBACKUP_LOCATION", f"{NFS_DIR}/db_backups/")
        },
    },
}

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
COLMAP_PATH = os.getenv("COLMAP_PATH", "colmap")  # CHANGEME: Path to colmap executable
GS_MAX_ITER = int(os.getenv("GS_MAX_ITER", "20000"))
ALPHA_CULL_THRESH = float(
    os.getenv("ALPHA_CULL_THRESH", "0.005")
)  # Threshold to delete translucent gaussians - lower values remove more (usually better quality)
MIN_MATCHED_IMAGES = int(os.getenv("MIN_MATCHED_IMAGES", "5"))
MIN_MATCH_RATIO = float(os.getenv("MIN_MATCH_RATIO", "0.10"))

# VGGT
VGGT_SCRIPT_PATH = "./tirtha/run_vggt.py"
VGGT_ENV_PATH = "../.venv"

# MR
# NOTE: Defaulting to Meshroom 2021 for now. 2023/25 will require further changes
ALICEVISION_DIRPATH = Path(
    os.getenv("ALICEVISION_DIRPATH", BASE_DIR / "bin21")
)  # CHANGEME:
# NOTE: See `Requirements` section in README.md
NSFW_MODEL_DIRPATH = BASE_DIR / "nn_models/nsfw_model/mobilenet_v2_140_224/"
MANIQA_MODEL_FILEPATH = BASE_DIR / "nn_models/MANIQA/ckpt_kadid10k.pt"

OBJ2GLTF_PATH = os.getenv("OBJ2GLTF_PATH", "obj2gltf")
GLTFPACK_PATH = os.getenv("GLTFPACK_PATH", "gltfpack")
MESHOPS_MIN_IMAGES = int(
    os.getenv("MESHOPS_MIN_IMAGES", "10")
)  # CHANGEME: Minimum number of images required to run meshops
MESHOPS_MAX_IMAGES = int(os.getenv("MESHOPS_MAX_IMAGES", "500"))
MESHOPS_CONTRIB_DELAY = float(
    os.getenv("MESHOPS_CONTRIB_DELAY", str(0.005))
)  # 18 seconds for testing | Keep >= 1 hour(s) - CHANGEME: time to wait before running meshops after a new contribution
FILE_UPLOAD_MAX_MEMORY_SIZE = int(
    os.getenv("FILE_UPLOAD_MAX_MEMORY_SIZE", str(10_485_760 * 2))
)  # 20 MiB (each file max - post compression)
DATA_UPLOAD_MAX_NUMBER_FILES = int(
    os.getenv("DATA_UPLOAD_MAX_NUMBER_FILES", "2000")
)  # 2_000 files # Max number of files per upload

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
FALLBACK_ARK_RESOLVER = os.getenv("FALLBACK_ARK_RESOLVER", "https://n2t.net")

## Email settings
# SMTP configuration for sending emails (e.g., new contributor notifications)
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")  # CHANGEME: Use your SMTP server
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() == "true"
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "")  # CHANGEME: Your email address
EMAIL_HOST_PASSWORD = os.getenv(
    "EMAIL_HOST_PASSWORD", ""
)  # CHANGEME: Your email password or app password
DEFAULT_FROM_EMAIL = os.getenv(
    "DEFAULT_FROM_EMAIL", ADMIN_MAIL
)  # CHANGEME: Default sender email
ADMINS = [
    (
        ADMIN_NAME,
        ADMIN_MAIL,
    ),  # CHANGEME: Admin email addresses
]
SERVER_EMAIL = os.getenv(
    "SERVER_EMAIL", ADMIN_MAIL
)  # CHANGEME: For server error emails

# List of contributor email addresses to ignore when sending new contribution notifications
CONTRIB_IGNORE_LIST = [
    ADMIN_MAIL,
    SERVER_EMAIL,
    *[e.strip() for e in os.getenv("CONTRIB_IGNORE_LIST", "").split(",") if e.strip()],
]

# Toggle for sending success emails to contributors when their contribution processing finishes
MAIL_CONTRIB_TOGGLE = os.getenv("MAIL_CONTRIB_TOGGLE", "False").lower() == "true"
