from pathlib import Path

from django.conf import settings

# Local imports
from .celery import app, crontab, get_task_logger
from .utils import Logger

cel_logger = get_task_logger(__name__)
LOG_DIR = Path(settings.LOG_DIR)
MESHOPS_CONTRIB_DELAY = settings.MESHOPS_CONTRIB_DELAY  # hours
BACKUP_INTERVAL = crontab(minute=0, hour=0, day_of_week=0)  # Every Sunday at 00:00
DBCLEANUP_INTERVAL = crontab(
    minute=0, hour=0, day_of_week=0
)  # Every week at 00:00 on Sunday


@app.task
def post_save_contrib_imageops(contrib_id: str) -> None:
    """
    Triggers `ImageOps`, when a `Contribution` instance is created & saved.

    """
    from .imageops import ImageOps

    # Check images
    cel_logger.info(
        f"post_save_contrib_imageops: Triggering ImageOps for contrib_id: {contrib_id}."
    )
    cel_logger.info(
        f"post_save_contrib_imageops: Checking images for contrib_id: {contrib_id}..."
    )
    iops = ImageOps(contrib_id=contrib_id)
    # FIXME: TODO: Till the VRAM + concurrency issue is fixed, skipping image checks.
    iops.check_images()
    # cel_logger.info(
    #     f"post_save_contrib_imageops: Finished checking images for contrib_id: {contrib_id}."
    # )
    cel_logger.info(
        f"Skipping image checks for contrib_id: {contrib_id} due to VRAM + concurrency issues. FIXME:"
    )

    # FIXME: TODO: MESHOPS_CONTRIB_DELAY = 0.1 (6 minutes), till the image checks are fixed.
    # Create mesh after MESHOPS_CONTRIB_DELAY hours

    cel_logger.info(
        f"post_save_contrib_imageops: Will trigger reconstruction pipelines for {contrib_id} after {MESHOPS_CONTRIB_DELAY} hours..."
    )
    recon_runner_task.apply_async(
        args=(contrib_id,), countdown=MESHOPS_CONTRIB_DELAY * 60 * 60
    )


@app.task
def recon_runner_task(contrib_id: str) -> None:
    """
    Triggers `MeshOps` & `GSOps`, when a `Run` instance is created.

    """
    from .workers import prerun_check, ops_runner

    cel_logger.info(
        f"recon_runner_task: Running prerun checks for contrib_id: {contrib_id}..."
    )
    chk, msg = prerun_check(contrib_id)
    cel_logger.info(f"recon_runner_task: {contrib_id} - {msg}")
    if chk:
        for op in ["GS", "aV"]:
            cel_logger.info(
                f"recon_runner_task: Triggering {op}Ops for contrib_id: {contrib_id}."
            )
            cel_logger.info(f"recon_runner_task: Running {op}Ops for {contrib_id}...")
            ops_runner(contrib_id=contrib_id, kind=op)
            cel_logger.info(
                f"recon_runner_task: Finished running {op}Ops for {contrib_id}."
            )


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
    - Removes contributors with no contributions for privacy reasons.

    """
    cln_logger = Logger(name="db_cleanup", log_path=LOG_DIR)
    cln_logger.info("db_cleanup_task: Cleaning up database...")


@app.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls backup_task() every BACKUP_INTERVAL.
    sender.add_periodic_task(BACKUP_INTERVAL, backup_task.s(), name="backup_task")

    # Calls db_cleanup_task() every DBCLEANUP_INTERVAL.
    sender.add_periodic_task(
        DBCLEANUP_INTERVAL, db_cleanup_task.s(), name="db_cleanup_task"
    )
