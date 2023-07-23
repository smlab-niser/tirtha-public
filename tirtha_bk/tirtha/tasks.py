from pathlib import Path

from django.conf import settings

# Local imports
from tirtha.models import Contributor

from .celery import app, crontab, get_task_logger
from .utils import Logger

cel_logger = get_task_logger(__name__)
LOG_DIR = Path(settings.LOG_DIR)
MESHOPS_CONTRIB_DELAY = settings.MESHOPS_CONTRIB_DELAY  # hours
BACKUP_INTERVAL = crontab(minute=0, hour=0)  # Every 24 hours at 00:00
DBCLEANUP_INTERVAL = crontab(
    minute=0, hour=0, day_of_week=0
)  # Every 1 week at 00:00 on Sunday


@app.task
def post_save_contrib_imageops(contrib_id):
    """
    Triggers `ImageOps`, when a `Contribution` instance is created & saved.

    """
    from .workers import ImageOps

    # Check images
    cel_logger.info(
        f"post_save_contrib_imageops: Triggering ImageOps for contrib_id: {contrib_id}."
    )
    cel_logger.info(
        f"post_save_contrib_imageops: Checking images for contrib_id: {contrib_id}..."
    )
    iops = ImageOps(contrib_id=contrib_id)
    iops.check_images()
    cel_logger.info(
        f"post_save_contrib_imageops: Finished checking images for contrib_id: {contrib_id}."
    )

    # Create mesh after MESHOPS_CONTRIB_DELAY hours
    cel_logger.info(
        f"post_save_contrib_imageops: Will trigger MeshOps for {contrib_id} after {MESHOPS_CONTRIB_DELAY} hours..."
    )
    mo_runner_task.apply_async(
        args=(contrib_id,), countdown=MESHOPS_CONTRIB_DELAY * 60 * 60
    )


@app.task
def mo_runner_task(contrib_id):
    """
    Triggers `MeshOps`, when a `Run` instance is created.

    """
    from .workers import mo_runner, prerun_check

    cel_logger.info(f"mo_runner_task: Triggering MeshOps for contrib_id: {contrib_id}.")
    cel_logger.info(
        f"mo_runner_task: Running prerun checks for contrib_id: {contrib_id}..."
    )
    chk, msg = prerun_check(contrib_id)
    cel_logger.info(f"mo_runner_task: {contrib_id} - {msg}")
    if chk:
        cel_logger.info(f"mo_runner_task: Running MeshOps for {contrib_id}...")
        mo_runner(contrib_id=contrib_id)
        cel_logger.info(f"mo_runner_task: Finished running MeshOps for {contrib_id}.")


@app.task
def backup_task():
    """
    Backs up the database & media files using django-dbbackup.

    """
    from django.core.management import call_command

    # Setup logger
    bak_logger = Logger(name="db_backup", log_path=LOG_DIR)

    # Backup database & media files
    bak_logger.info("backup_task: Backing up database & media files...")
    cel_logger.info("backup_task: Backing up database & media files...")
    call_command("dbbackup")  # LATE_EXP: Add other options
    bak_logger.info("backup_task: Backed up database.")
    call_command("mediabackup")  # LATE_EXP: Add other options
    bak_logger.info("backup_task: Backed up media files.")
    cel_logger.info("backup_task: Backed up database & media files.")


@app.task
def db_cleanup_task():
    """
    Cleans up the database.
    - Removes contributors with no contributions.

    """
    cln_logger = Logger(name="db_cleanup", log_path=LOG_DIR)
    cln_logger.info("db_cleanup_task: Cleaning up database...")
    Contributor.objects.filter(contributions__isnull=True).delete()
    cln_logger.info("db_cleanup_task: Cleaned up database.")


@app.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls backup_task() every BACKUP_INTERVAL.
    sender.add_periodic_task(BACKUP_INTERVAL, backup_task.s(), name="backup_task")

    # Calls db_cleanup_task() every DBCLEANUP_INTERVAL.
    sender.add_periodic_task(
        DBCLEANUP_INTERVAL, db_cleanup_task.s(), name="db_cleanup_task"
    )
