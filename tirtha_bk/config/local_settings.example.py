# NOTE: Rename to local_settings.py, and...
# ...place this file in ./tirtha-public/tirtha_bk/tirtha_bk/
"""
Fields marked CHANGEME: need to be changed before deployment

"""
import os
from pathlib import Path

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "<your_secret_key>"  # CHANGEME:

TIME_ZONE = "<your_time_zone>"  # CHANGEME: if needed

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True  # NOTE: Set to False in production  # CHANGEME:
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin-allow-popups"
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "<your_ip_address>"]  # CHANGEME:

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

PROD_DIR = "<your_prod_dir>"  # Short term storage for current runs # CHANGEME:
LOG_DIR = f"{PROD_DIR}logs"
NFS_DIR = "<your_nfs_dir>"  # Long term storage for old runs # CHANGEME: Does not need to use NFS
ARCHIVE_ROOT = f"{NFS_DIR}"

# Static files
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]  # JS, CSS, images, favicon
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(PROD_DIR, "static")

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = f"{PROD_DIR}media"
# STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default attributes used to create default mesh & contributor
# NOTE: You will need to create the default mesh & contributor manually before running the server for the first time.
# TODO: FIXME: upload a default glb file + extra info and point to that in the repo. Maybe set default mesh as hidden and completed.
DEFAULT_MESH_NAME = "<your_default_mesh_name>"  # Default mesh name to use while setting up | Will be shown on homepage # CHANGEME:
DEFAULT_MESH_ID = "<your_default_mesh_id>"  # CHANGEME: NOTE: Must be a ShortUUID string of length 16. You can use https://shortunique.id/ to generate one.
ADMIN_NAME = "<your_admin_name>"  # CHANGEME:
ADMIN_MAIL = "<your_admin_mail>"  # CHANGEME:

# Sign in with Google
GOOGLE_CLIENT_ID = "<your_google_client_id>"  # CHANGEME:
COOKIE_EXPIRE_TIME = 3600  # 1 hour
SESSION_COOKIE_SAMESITE = "Strict"
SESSION_COOKIE_SECURE = True

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases
# NOTE: For help, see https://www.digitalocean.com/community/tutorials/how-to-set-up-django-with-postgres-nginx-and-gunicorn-on-ubuntu-22-04
DB_NAME = "<your_db_name>"  # CHANGEME:
DB_USER = "<your_db_user>"  # CHANGEME:
DB_PWD = "<your_db_pwd>"  # CHANGEME:
DB_HOST = "localhost"  # CHANGEME: if needed
DB_PORT = ""

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
DBBACKUP_STORAGE_OPTIONS = {"location": f"{NFS_DIR}/db_backups/"}

## RabbitMQ + Celery
RMQ_USER = "<your_rmq_user>"  # CHANGEME:
RMQ_PWD = "<your_rmq_pwd>"  # CHANGEME:
RMQ_VHOST = "<your_rmq_vhost>"  # CHANGEME: NOTE: You can use the default vhost (`/`) if you want.
CELERY_BROKER_URL = f"pyamqp://{RMQ_USER}:{RMQ_PWD}@localhost/{RMQ_VHOST}"
CELERY_TASK_ACKS_LATE = True  # To prevent tasks from being lost
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # Disable prefetching
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1  # NOTE: For memory release: https://stackoverflow.com/questions/17541452/celery-does-not-release-memory
CELERY_BROKER_CONNECTION_RETRY = False
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_CONNECTION_MAX_RETRIES = 10

## Worker-related settings
# NOTE: Defaulting to Meshroom 2021 for now. 2023 will require further changes
ALICEVISION_DIRPATH = BASE_DIR / "bin21"
NSFW_MODEL_DIRPATH = (
    BASE_DIR / "nn_models/nsfw_model/mobilenet_v2_140_224/"
)  # NOTE: See `Requirements` section in README.md
OBJ2GLTF_PATH = "obj2gltf"  # NOTE: Ensure the binary is in system PATH
GLTFPACK_PATH = "gltfpack"  # NOTE: Ensure the binary is in system PATH
MESHOPS_MIN_IMAGES = 20  # Minimum number of images required to run meshops
MESHOPS_CONTRIB_DELAY = (
    1  # hour(s) - time to wait before running meshops after a new contribution
)

## ARK settings
BASE_URL = "<your_base_url>"  # CHANGEME: NOTE: No trailing '/' | e.g., https://tirtha.niser.ac.in (Use your own)
ARK_NAAN = int("<your_ark_naan>")  # CHANGEME: Integer
ARK_SHOULDER = "<your_ark_shoulder>"  # CHANGEME:
FALLBACK_ARK_RESOLVER = "https://n2t.net"


FILE_UPLOAD_MAX_MEMORY_SIZE = (
    10485760  # CHANGEME: 10 MiB (each file max size - post compression)
)
DATA_UPLOAD_MAX_NUMBER_FILES = 100  # CHANGEME: Max number of files per upload
