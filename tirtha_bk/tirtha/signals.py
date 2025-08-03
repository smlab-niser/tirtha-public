import shutil
import os
from pathlib import Path

from django.conf import settings
from django.db.models.signals import post_delete, post_migrate, post_save, pre_save
from django.dispatch import receiver

# Local imports
from .models import Contribution, Contributor, Image, Mesh, Run

STATIC = Path(settings.STATIC_ROOT)
MEDIA = Path(settings.MEDIA_ROOT)
DEFAULT_MESH_NAME = settings.DEFAULT_MESH_NAME
DEFAULT_MESH_ID = settings.DEFAULT_MESH_ID
ADMIN_NAME = settings.ADMIN_NAME
ADMIN_MAIL = settings.ADMIN_MAIL


@receiver(post_migrate)
def post_migrate_create_defaults(sender, **kwargs):
    """
    Creates the default mesh & contributor post migration.

    """
    if (
        sender.name == "tirtha"
        and Mesh.objects.count() == 0
        and Contributor.objects.count() == 0
    ):
        mesh_ID = DEFAULT_MESH_ID

        default_desc = "This is the meditation center, atop a small hill, inside the NISER campus at Khordha, Odisha."

        static_paths = [
            STATIC / f"models/{mesh_ID}/avcache",
            STATIC / f"models/{mesh_ID}/published",
        ]
        for static_path in static_paths:
            if not static_path.exists():
                static_path.mkdir(parents=True)

        # Copy default mesh file to STATIC / models / mesh_ID / published
        source = STATIC / f"{mesh_ID}"
        fname = f"{mesh_ID}__default.glb"
        src = source / f"{fname}"
        dest = STATIC / f"models/{mesh_ID}/published/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, dest)

        # Copy default mesh thumbnail and preview images from STATIC to MEDIA
        source /= f"to_media"
        srcs = [source / f"{mesh_ID}_thumb.jpg", source / f"{mesh_ID}_prev.jpg"]
        dest = MEDIA / f"models/{mesh_ID}/"
        dest.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            if(os.path.exists(src)):
                shutil.copy2(src, dest)

        # Create default mesh - shown on homepage
        mesh, _ = Mesh.objects.get_or_create(
            ID=mesh_ID, name=DEFAULT_MESH_NAME, hidden=True
        )
        mesh.description = default_desc
        mesh.preview = f"models/{mesh_ID}/{mesh_ID}_prev.jpg"
        mesh.thumbnail = f"models/{mesh_ID}/{mesh_ID}_thumb.jpg"
        mesh.minObsAng = 70
        if(os.path.exists(mesh.preview) and os.path.exists(mesh.thumbnail)):
            mesh.save()
        else:
            pass
        # Create default contributor
        Contributor.objects.create(name=ADMIN_NAME, email=ADMIN_MAIL)


# Connect the signal
post_migrate.connect(post_migrate_create_defaults)


@receiver(post_save, sender=Mesh)
def post_save_mesh(sender, instance, **kwargs):
    """
    Creates corresponding directories in filesystem

    """
    mesh_ID = instance.ID

    # Create these folders in MEDIA, if they don't exist
    to_create = ["images", "images/nsfw", "images/good", "images/bad"]
    for folder in to_create:
        path = MEDIA / f"models/{mesh_ID}/{folder}"
        if not path.exists():
            path.mkdir(parents=True)  # Makes both these & mesh_ID folders

    static_paths = [
        STATIC / f"models/{mesh_ID}/avcache",
        STATIC / f"models/{mesh_ID}/published",
    ]
    for static_path in static_paths:
        if not static_path.exists():
            static_path.mkdir(parents=True)  # Makes both cache & mesh_ID folders

    # LATE_EXP: if mesh.status == "Live", then:
    # 1. Create preview image
    # 2. Assign it to mesh.preview
    # FIXME: Do this per `Run` instead.


@receiver(post_delete, sender=Mesh)
def post_del_mesh(sender, instance, **kwargs):
    """
    Deletes corresponding directories from filesystem post Mesh deletion.

    """
    mesh_ID = instance.ID
    src = MEDIA / f"models/{mesh_ID}"
    dest = STATIC / f"models/{mesh_ID}"

    shutil.rmtree(src)
    shutil.rmtree(dest)


@receiver(post_delete, sender=Image)
def post_del_image(sender, instance, **kwargs):
    """
    Deletes the corresponding contribution if no images are left in it.
    # NOTE: Deletion from MEDIA is handled by django-cleanup

    """
    Contribution.objects.filter(images__isnull=True).delete()


@receiver(pre_save, sender=Image)
def pre_save_image(sender, instance, **kwargs):
    """
    Moves image to appropriate folder - nsfw, good, bad, when Image.label is changed.

    """
    if instance.pk:
        old_instance = Image.objects.get(pk=instance.pk)

        if instance.label != old_instance.label:
            image_root = f"models/{instance.contribution.mesh.ID}/images/"
            src = MEDIA / instance.image.name
            fname = instance.image.name.split("/")[-1]
            if not instance.label:
                dest = MEDIA / image_root / f"{fname}"
            else:
                dest = MEDIA / image_root / f"{instance.label}/{fname}"
            if src != dest:
                shutil.move(src, dest)  # Move image
                instance.image.name = (
                    f"{image_root}{instance.label}/{fname}"  # Update path in DB
                )


@receiver(post_save, sender=Run)
def post_save_run(sender, instance, **kwargs):
    """
    Creates the run folder, when a `Run` instance is created.

    """
    run_path = STATIC / f"models/{instance.directory}"

    if not run_path.exists():
        run_path.mkdir(parents=True)


@receiver(post_delete, sender=Run)
def post_del_run(sender, instance, **kwargs):
    """
    Deletes the run folder, when a `Run` instance is deleted.

    """
    run_dir = STATIC / f"models/{instance.directory}"

    # Delete only if the run is not archived
    # NOTE: These runs had succeeded, so ARKs were generated. Have to keep them.
    if run_dir.exists() and instance.status != "Archived":
        shutil.rmtree(run_dir)
