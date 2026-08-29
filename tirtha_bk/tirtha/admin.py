import os
import re
import time
import shutil
import tempfile
import zipfile
import threading
import logging
from datetime import datetime, timezone

from django import forms
from django.conf import settings
from django.contrib import admin, messages
from django.urls import reverse
from django.utils.html import mark_safe
from django.utils.translation import ngettext
from django.http import HttpResponse, Http404, StreamingHttpResponse
from django.shortcuts import redirect

# Local imports
from .models import ARK, Contribution, Contributor, Image, Mesh, Run
from .tasks import post_save_contrib_imageops, recon_runner_task


# Logger setup
# Logging
ADMIN_LOG_LOCATION = settings.ADMIN_LOG_LOCATION

logging.basicConfig(
    level=logging.NOTSET,
    format="%(asctime)s %(levelname)s %(message)s",
    filename=ADMIN_LOG_LOCATION,
)


class ContributionsInline(admin.TabularInline):
    def contribution_ts(self, obj):
        return obj.contributed_at

    contribution_ts.short_description = "Contribution Timestamp"

    def contribution_link(self, obj):
        url = reverse("admin:tirtha_contribution_change", args=[obj.ID])
        return mark_safe(f'<a href="{url}">{obj.ID}</a>')

    contribution_link.short_description = "Contribution Link"

    model = Contribution
    readonly_fields = (
        "contribution_ts",
        "contribution_link",
    )  # FIXME: Add "processed"
    fields = ("ID", "contribution_ts", "contribution_link", "processed")
    extra = 0
    max_num = 0
    can_delete = True


class ContributionInlineMesh(ContributionsInline):
    def contributor_email(self, obj):
        return obj.contributor.email

    contributor_email.short_description = "Contributor Email"

    readonly_fields = ContributionsInline.readonly_fields + ("contributor_email",)
    fields = ContributionsInline.fields + ("contributor_email",)


class ContributionInlineContributor(ContributionsInline):
    def mesh_id(self, obj):
        return obj.mesh.verbose_id

    mesh_id.short_description = "Mesh ID (Verbose)"

    readonly_fields = ContributionsInline.readonly_fields + ("mesh_id",)
    fields = ContributionsInline.fields + ("mesh_id",)


class RunInlineMesh(admin.TabularInline):
    model = Run
    readonly_fields = ("ID", "ark", "status", "started_at", "ended_at")
    fields = ("ID", "ark", "status", "started_at", "ended_at")
    extra = 0
    max_num = 0
    can_delete = False


@admin.register(Mesh)
class MeshAdmin(admin.ModelAdmin):
    def get_preview(self, obj):
        return mark_safe(
            f'<img src="{obj.preview.url}" alt="{str(obj.verbose_id)}" style="width: 400px; height: 400px">'
        )

    get_preview.short_description = "Preview"

    def get_thumbnail(self, obj):
        return mark_safe(
            f'<img src="{obj.thumbnail.url}" alt="{str(obj.verbose_id)}" style="width: 400px; height: 400px">'
        )

    get_thumbnail.short_description = "Thumbnail"

    def mesh_id_verbose(self, obj):
        return obj.verbose_id

    mesh_id_verbose.short_description = "ID (Verbose)"

    def contrib_count(self, obj):
        return obj.contributions.count()

    contrib_count.short_description = "Contribution Count"

    def image_count(self, obj):
        return Image.objects.filter(contribution__mesh=obj).count()

    image_count.short_description = "Total Image Count"

    @admin.action(description="Mark selected sites as completed")
    def mark_completed(self, request, queryset):
        updated = queryset.update(completed=True)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as completed.",
                "%d sites were successfully marked as completed.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected sites as incomplete")
    def mark_incomplete(self, request, queryset):
        updated = queryset.update(completed=False)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as incomplete.",
                "%d sites were successfully marked as incomplete.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected sites as hidden")
    def mark_hidden(self, request, queryset):
        updated = queryset.update(hidden=True)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as hidden.",
                "%d sites were successfully marked as hidden.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected sites as not hidden")
    def mark_not_hidden(self, request, queryset):
        updated = queryset.update(hidden=False)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as not hidden.",
                "%d sites were successfully marked as not hidden.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    actions = [
        "mark_completed",
        "mark_incomplete",
        "mark_hidden",
        "mark_not_hidden",
        "download_images_zip_meshes",
    ]
    readonly_fields = (
        "ID",
        "download_link",
        "verbose_id",
        "created_at",
        "updated_at",
        "reconstructed_at",
        "get_preview",
        "get_thumbnail",
    )
    fieldsets = (
        (
            "Mesh Details",
            {
                "fields": (
                    (
                        "ID",
                        "verbose_id",
                        "download_link",
                    ),
                    ("created_at", "updated_at", "reconstructed_at"),
                    ("status", "completed", "hidden"),
                    ("name", "country", "state", "district"),
                    "description",
                    ("center_image", "denoise"),
                    (
                        "rotaZ",
                        "rotaX",
                        "rotaY",
                        "minObsAng",
                        "orientMesh",
                    ),  # Mimicking <model-viewer> attributes (ZXY)
                    ("thumbnail", "get_thumbnail"),
                    ("preview", "get_preview"),
                )
            },
        ),
    )
    list_filter = (
        "status",
        "completed",
        "hidden",
    )
    list_display = (
        "ID",
        "mesh_id_verbose",
        "name",
        "reconstructed_at",
        "status",
        "completed",
        "hidden",
        "contrib_count",
        "image_count",
        "get_thumbnail",
        "download_link",
    )
    list_per_page = 25
    inlines = [ContributionInlineMesh]  # , RunInlineMesh FIXME: Error while saving

    def get_urls(self):
        from django.urls import path

        urls = super().get_urls()
        my_urls = [
            path(
                "<str:object_id>/download_images/",
                self.admin_site.admin_view(self.download_images_view),
                name="tirtha_mesh_download_images",
            )
        ]
        return my_urls + urls

    def download_images_view(self, request, object_id):
        # Per-mesh download via admin link
        try:
            mesh = Mesh.objects.get(ID=object_id)
        except Mesh.DoesNotExist:
            raise Http404("Mesh not found")

        # Build a queryset of contributions for this mesh and call the same zipping logic
        contrib_qs = mesh.contributions.all()
        # reuse ContributionAdmin-like zipping logic but scoped to these contributions
        # Build an in-memory zip: contribid/<label>/file

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        verboseid_s = _sanitize(mesh.verbose_id)[:50]
        fname = f"{verboseid_s}.zip"

        # Collect files to zip: (src_path, arcname)
        file_tuples = []
        for contrib in contrib_qs:
            for img in contrib.images.all():
                try:
                    img_path = img.image.path
                except Exception:
                    continue
                if not os.path.exists(img_path):
                    continue
                label = img.label if img.label else "unknown"
                arcname = os.path.join(
                    str(contrib.ID), label, os.path.basename(img_path)
                )
                file_tuples.append((img_path, arcname))

        # Write ZIP to temp file and wait for completion, then stream
        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()

        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for src, arc in file_tuples:
                        try:
                            zf.write(src, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {src} to mesh zip {fname}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating mesh zip {fname}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()

        # Wait for writer to finish (ensures valid ZIP)
        writer_done.wait()

        def _stream():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(_stream(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            logging.info(
                f"ADMIN MESH ZIP DOWNLOAD by {request.user} - mesh={verboseid_s} filename={fname}"
            )
        except Exception:
            pass

        return resp

    def download_link(self, obj):
        try:
            url = reverse("admin:tirtha_mesh_download_images", args=[obj.ID])
            return mark_safe(
                f'<a class="button" href="{url}">Download All Contributions</a>'
            )
        except Exception:
            return ""

    download_link.short_description = "Download"

    @admin.action(description="Download images as ZIP for selected meshes")
    def download_images_zip_meshes(self, request, queryset):
        meshes = list(queryset)

        if not meshes:
            self.message_user(request, "No meshes selected.", level=messages.WARNING)
            return

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        mesh_verboseids = [_sanitize(m.verbose_id)[:50] for m in meshes]

        if len(meshes) == 1:
            # single mesh: structure contribid/label/file
            mesh = meshes[0]
            verboseid_s = mesh_verboseids[0]
            fname = f"{verboseid_s}.zip"
        else:
            ids_short = "_".join([v[:4] for v in mesh_verboseids])
            fname = f"multiple_meshes_{ids_short}.zip"

        # Collect files into file_tuples
        file_tuples = []
        for mesh in meshes:
            is_multi = len(meshes) > 1
            for contrib in mesh.contributions.all():
                for img in contrib.images.all():
                    try:
                        img_path = img.image.path
                    except Exception:
                        continue
                    if not os.path.exists(img_path):
                        continue
                    label = img.label if img.label else "unknown"
                    if is_multi:
                        mesh_vid = _sanitize(mesh.verbose_id)[:50]
                        arcname = os.path.join(
                            mesh_vid, str(contrib.ID), label, os.path.basename(img_path)
                        )
                    else:
                        arcname = os.path.join(
                            str(contrib.ID), label, os.path.basename(img_path)
                        )
                    file_tuples.append((img_path, arcname))

        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()
        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for src, arc in file_tuples:
                        try:
                            zf.write(src, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {src} to mesh bulk zip {fname}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating mesh bulk zip {fname}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        writer_done.wait()

        def _stream():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(_stream(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            mesh_ids = [str(m.ID) for m in meshes]
            logging.info(
                f"ADMIN MESH ZIP DOWNLOAD by {request.user} - meshes={mesh_ids} filename={fname}"
            )
        except Exception:
            pass

        return resp


@admin.register(Contributor)
class ContributorAdmin(admin.ModelAdmin):
    def contrib_count(self, obj):
        return obj.contributions.count()

    contrib_count.short_description = "Contribution Count"

    def image_count(self, obj):
        return Image.objects.filter(contribution__contributor=obj).count()

    image_count.short_description = "Total Image Count"

    @admin.action(description="Activate selected contributors")
    def activate_contributors(self, request, queryset):
        updated = queryset.update(active=True)
        self.message_user(
            request,
            ngettext(
                "%d contributor was successfully activated.",
                "%d contributors were successfully activated.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Deactivate selected contributors")
    def deactivate_contributors(self, request, queryset):
        updated = queryset.update(active=False)
        self.message_user(
            request,
            ngettext(
                "%d contributor was successfully deactivated.",
                "%d contributors were successfully deactivated.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Ban selected contributors")
    def ban_contributors(self, request, queryset):
        updated = queryset.update(banned=True)
        self.message_user(
            request,
            ngettext(
                "%d contributor was successfully banned.",
                "%d contributors were successfully banned.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Unban selected contributors")
    def unban_contributors(self, request, queryset):
        updated = queryset.update(banned=False)
        self.message_user(
            request,
            ngettext(
                "%d contributor was successfully unbanned.",
                "%d contributors were successfully unbanned.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    actions = [
        activate_contributors,
        deactivate_contributors,
        ban_contributors,
        unban_contributors,
    ]
    readonly_fields = (
        "ID",
        "created_at",
        "updated_at",
    )
    fieldsets = (
        (
            "Contributor Details",
            {
                "fields": (
                    "ID",
                    ("created_at", "updated_at"),
                    ("name", "email"),
                    "active",
                    "banned",
                    "ban_reason",
                )
            },
        ),
    )
    inlines = [ContributionInlineContributor]
    list_filter = (
        "active",
        "banned",
    )
    list_display = (
        "ID",
        "name",
        "email",
        "created_at",
        "updated_at",
        "contrib_count",
        "image_count",
        "active",
        "banned",
    )
    list_per_page = 50


class ImageInlineContribution(admin.TabularInline):
    """
    Shows images in the Contribution admin page

    """

    def get_image(self, obj):
        return mark_safe(
            f'<img src="{obj.image.url}" style="width: 400px; height: 400px">'
        )

    get_image.short_description = "Preview"

    def image_link(self, obj):
        url = reverse("admin:tirtha_image_change", args=[obj.ID])
        return mark_safe(f'<a href="{url}">{obj.ID}</a>')

    image_link.short_description = "Image Link"

    def image_label(self, obj):
        return obj.label.upper()

    image_label.short_description = "Label"

    model = Image
    readonly_fields = ("get_image", "image_link", "image_label")
    fields = (
        "image_link",
        "get_image",
        "image_label",
    )
    extra = 0
    max_num = 0
    can_delete = True


@admin.register(Contribution)
class ContributionAdmin(admin.ModelAdmin):
    def mesh_id_verbose(self, obj):
        return obj.mesh.verbose_id

    mesh_id_verbose.short_description = "Mesh ID (Verbose)"

    def mesh_name(self, obj):
        return obj.mesh.name

    mesh_name.short_description = "Mesh Name"

    def image_count(self, obj):
        return obj.images.count()

    image_count.short_description = "Image Count"

    def images_good_count(self, obj):
        return obj.images.filter(label="good").count()

    images_good_count.short_description = "Good Image Count"

    def get_urls(self):
        from django.urls import path

        urls = super().get_urls()
        my_urls = [
            path(
                "<str:object_id>/download_images/",
                self.admin_site.admin_view(self.download_images_view),
                name="tirtha_contribution_download_images",
            )
        ]
        return my_urls + urls

    def download_images_view(self, request, object_id):
        # Single contribution download via a per-object admin link
        try:
            contrib = Contribution.objects.get(ID=object_id)
        except Contribution.DoesNotExist:
            raise Http404("Contribution not found")

        return self.download_images_zip(
            request, Contribution.objects.filter(ID=contrib.ID)
        )

    def download_link(self, obj):
        try:
            url = reverse("admin:tirtha_contribution_download_images", args=[obj.ID])
            return mark_safe(f'<a class="button" href="{url}">Download ZIP</a>')
        except Exception:
            return ""

    download_link.short_description = "Download"

    @admin.action(description="Mark selected contributions as processed")
    def mark_processed(self, request, queryset):
        updated = queryset.update(processed=True)
        self.message_user(
            request,
            ngettext(
                "%d contribution was successfully marked as processed.",
                "%d contributions were successfully marked as processed.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(
        description="Trigger ImageOps & all reconstructions for selected contributions"
    )
    def trigger_imageops(self, request, queryset):
        updated = queryset.update(processed=False)
        for obj in queryset:
            post_save_contrib_imageops.delay(
                str(obj.ID), recons_type="all"
            )  # This triggers ImageOps, which in turn triggers GSOPs or MeshOps
            logging.info(
                f"ADMIN -- ImageOps & all reconstructions successfully triggered for {obj.ID}."
            )
        self.message_user(
            request,
            ngettext(
                "ImageOps & all reconstructions successfully triggered for %d contribution.",
                "ImageOps & all reconstructions successfully triggered for %d contributions.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Trigger aVOps for selected contributions")
    def trigger_aVOps(self, request, queryset):
        count = queryset.count()
        for obj in queryset:
            recon_runner_task.delay(str(obj.ID), recons_type="aV")
            logging.info(f"ADMIN -- aVOps successfully triggered for {obj.ID}.")
        self.message_user(
            request,
            ngettext(
                "aVOps successfully triggered for %d contribution.",
                "aVOps successfully triggered for %d contributions.",
                count,
            )
            % count,
            messages.SUCCESS,
        )

    @admin.action(description="Trigger GSOps for selected contributions")
    def trigger_GSOps(self, request, queryset):
        count = queryset.count()
        for obj in queryset:
            recon_runner_task.delay(str(obj.ID), recons_type="GS")
            logging.info(f"ADMIN -- GSOps successfully triggered for {obj.ID}.")
        self.message_user(
            request,
            ngettext(
                "GSOps successfully triggered for %d contribution.",
                "GSOps successfully triggered for %d contributions.",
                count,
            )
            % count,
            messages.SUCCESS,
        )

    @admin.action(description="Download images as ZIP for selected contribution")
    def download_images_zip(self, request, queryset):
        contribs = list(queryset)

        if not contribs:
            self.message_user(
                request,
                "No contributions selected.",
                level=messages.WARNING,
            )
            return

        # Determine mesh name and verbose id. If contributions span meshes use fallback names
        mesh_set = {str(c.mesh.ID): c.mesh for c in contribs}
        if len(mesh_set) == 1:
            mesh = next(iter(mesh_set.values()))
            mesh_name = mesh.name
            verboseid = mesh.verbose_id
        else:
            mesh_name = "multiple_sites"
            verboseid = "multiple"

        # Sanitize mesh_name and verboseid for filenames
        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        mesh_name_s = _sanitize(mesh_name)[:100]
        verboseid_s = _sanitize(verboseid)[:50]

        if len(contribs) == 1:
            cid = str(contribs[0].ID)
            fname = f"{verboseid_s}_{cid[:8]}.zip"
        else:
            ids_short = "_".join([str(c.ID)[:4] for c in contribs])
            fname = f"{verboseid_s}_{ids_short}.zip"

        # Collect files and arc names
        file_tuples = []
        for contrib in contribs:
            for img in contrib.images.all():
                try:
                    img_path = img.image.path
                except Exception:
                    continue
                if not os.path.exists(img_path):
                    continue
                label = img.label if img.label else "unknown"
                if len(mesh_set) > 1:
                    mesh_vid = _sanitize(contrib.mesh.verbose_id)[:50]
                    arcname = os.path.join(
                        mesh_vid, str(contrib.ID), label, os.path.basename(img_path)
                    )
                else:
                    arcname = os.path.join(
                        str(contrib.ID), label, os.path.basename(img_path)
                    )
                file_tuples.append((img_path, arcname))

        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()
        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for src, arc in file_tuples:
                        try:
                            zf.write(src, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {src} to contrib zip {fname}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating contrib zip {fname}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        writer_done.wait()

        def _stream():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(_stream(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            ids_list = [str(c.ID) for c in contribs]
            logging.info(
                f"ADMIN ZIP DOWNLOAD by {request.user} - mesh={mesh_name_s} verboseid={verboseid_s} contribs={ids_list} filename={fname}"
            )
        except Exception:
            # Logging should not break the admin action
            pass

        return resp

    actions = [
        mark_processed,
        trigger_imageops,
        trigger_aVOps,
        trigger_GSOps,
        download_images_zip,
    ]
    readonly_fields = (
        "ID",
        "mesh",
        "contributor",
        "contributed_at",
        # "processed",
        "processed_at",
        "download_link",
        "image_count",
        "images_good_count",
    )
    fields = (
        "ID",
        "contributed_at",
        "mesh",
        "contributor",
        "processed",
        "processed_at",
        "download_link",
        "image_count",
        "images_good_count",
    )
    list_filter = (
        "processed",
        "mesh",
    )
    list_display = (
        "ID",
        "contributed_at",
        "mesh_name",
        "contributor",
        "image_count",
        "images_good_count",
        "processed_at",
        "processed",
        "download_link",
    )
    list_per_page = 50
    inlines = [
        ImageInlineContribution,
    ]


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    def note(self, obj):
        return (
            "PLEASE USE THE WEB INTERFACE TO ADD IMAGES.\nALSO, USE `Label` FOR MANUAL MODERATION.\n"
            + "ADD A REMARK IN `Remark` IF YOU ARE MANUALLY CHANGING THE LABEL."
        )

    def get_thumbnail(self, obj):
        return mark_safe(
            f'<img src="{obj.image.url}" style="width: 400px; height: 400px">'
        )

    get_thumbnail.short_description = "Preview"

    def get_mesh_id_verbose(self, obj):
        return obj.contribution.mesh.verbose_id

    get_mesh_id_verbose.short_description = "Mesh ID (Verbose)"

    def get_contributor_link(self, obj):
        url = reverse(
            "admin:tirtha_contributor_change", args=[obj.contribution.contributor.ID]
        )
        return mark_safe(f'<a href="{url}">{obj.contribution.contributor.name}</a>')

    get_contributor_link.short_description = "Contributor Link"

    @admin.action(description="Mark selected images as Good")
    def mark_good(self, request, queryset):
        updated = queryset.update(
            label="good"
        )  # FIXME: Not working because queryset.update does not trigger pre_save/post_save signals
        # https://stackoverflow.com/questions/1693145/django-signal-on-queryset-update/ FIXME:
        self.message_user(
            request,
            ngettext(
                "%d image was successfully marked as Good.",
                "%d images were successfully marked as Good.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected images as Bad")
    def mark_bad(self, request, queryset):
        updated = queryset.update(label="bad")
        self.message_user(
            request,
            ngettext(
                "%d image was successfully marked as Bad.",
                "%d images were successfully marked as Bad.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected images as NSFW")
    def mark_nsfw(self, request, queryset):
        updated = queryset.update(label="nsfw")
        self.message_user(
            request,
            ngettext(
                "%d image was successfully marked as NSFW.",
                "%d images were successfully marked as NSFW.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    actions = [mark_good, mark_bad, mark_nsfw]
    readonly_fields = (
        "ID",
        "contribution",
        "created_at",
        "image",
        "note",
        "get_thumbnail",
        "get_mesh_id_verbose",
        "get_contributor_link",
    )
    fieldsets = (
        (
            "Image Details",
            {
                "fields": (
                    ("note"),
                    ("ID"),
                    ("get_mesh_id_verbose"),
                    ("get_contributor_link"),
                    ("contribution"),
                    ("created_at"),
                    ("image", "get_thumbnail"),
                    ("label"),
                    ("remark"),
                )
            },
        ),
    )
    list_filter = ("label",)
    list_display = ("ID", "created_at", "contribution", "label", "get_thumbnail")
    list_per_page = 200


class ImageInlineRun(admin.TabularInline):
    """
    Shows images in the Run admin page

    """

    # def get_image(self, obj): # FIXME:
    #     return mark_safe(
    #         f'<img src="{obj.url}" style="width: 400px; height: 400px">'
    #     )

    # get_image.short_description = "Preview"

    def image_link(self, obj):
        url = reverse("admin:tirtha_image_change", args=[obj.image.ID])
        return mark_safe(f'<a href="{url}">{obj.image.ID}</a>')

    image_link.short_description = "Image Link"

    model = Run.images.through
    readonly_fields = ("image_link",)
    fields = ("image_link",)
    extra = 0
    can_delete = False


class ContributorInlineRun(admin.TabularInline):
    """
    Shows contributors in the Run admin page

    """

    model = Run.contributors.through
    extra = 0


class RunReplaceForm(forms.ModelForm):
    replacement = forms.FileField(
        required=False,
        label="Replace final output",
        help_text="Upload .gltf/.glb for MeshOps (aV) or .splat for GSOps. Staff or superusers only.",
    )

    class Meta:
        model = Run
        fields = []


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    form = RunReplaceForm

    def mesh_id_verbose(self, obj):
        return obj.mesh.verbose_id

    mesh_id_verbose.short_description = "Mesh ID (Verbose)"

    def image_count(self, obj):
        return obj.images.count()

    image_count.short_description = "Image Count"

    readonly_fields = (
        "ID",
        "ark",
        "run_links",
        "mesh_id_verbose",
        "kind",
        "started_at",
        "ended_at",
        "image_count",
        "status",
        "directory",
        "notes",
        "download_link",
    )
    fieldsets = (
        (
            "Run Metadata",
            {
                "fields": (
                    ("ID", "ark"),
                    "mesh_id_verbose",
                    "kind",
                    ("started_at", "ended_at"),
                    "directory",
                    "image_count",
                    (
                        "status",
                        "hidden",
                    ),
                    "notes",
                )
            },
        ),
        (
            "Editable / Viewer Settings",
            {
                "fields": (
                    (
                        "download_link",
                        "run_links",
                        "replacement",
                    ),
                    # <model-viewer>'s orientation
                    (
                        "rotaZ",
                        "rotaX",
                        "rotaY",
                    ),
                    # GS Viewer Settings
                    (
                        "initCamPosX",
                        "initCamPosY",
                        "initCamPosZ",
                    ),
                    (
                        "initCamLookAtX",
                        "initCamLookAtY",
                        "initCamLookAtZ",
                    ),
                    (
                        "camUpX",
                        "camUpY",
                        "camUpZ",
                    ),
                    (
                        "sphDegree",
                        "focalAdjustment",
                        "antialiased",
                    ),
                )
            },
        ),
    )
    list_filter = ("status", "kind")
    list_display = (
        "ID",
        "mesh_id_verbose",
        "kind",
        "image_count",
        "status",
        "started_at",
        "ark",
        "download_link",
    )
    list_per_page = 50
    inlines = [
        ContributorInlineRun
    ]  # ImageInlineRun, FIXME: Too many images lead to 400 (Bad Request)

    def get_urls(self):
        from django.urls import path

        urls = super().get_urls()
        my_urls = [
            path(
                "<str:object_id>/download_output/",
                self.admin_site.admin_view(self.download_output_view),
                name="tirtha_run_download_output",
            ),
            path(
                "<str:object_id>/download_run_full/",
                self.admin_site.admin_view(self.download_run_full_view),
                name="tirtha_run_download_full",
            ),
            path(
                "<str:object_id>/replace_output/",
                self.admin_site.admin_view(self.replace_output_view),
                name="tirtha_run_replace_output",
            ),
        ]
        return my_urls + urls

    def download_output_view(self, request, object_id):
        # Download only the final published output for the run (published folder)
        try:
            run = Run.objects.get(ID=object_id)
        except Run.DoesNotExist:
            raise Http404("Run not found")

        # Only allow replacement on Archived runs
        if run.status != "Archived":
            self.message_user(
                request,
                "Replacement allowed only for Archived runs.",
                level=messages.ERROR,
            )
            from django.core.exceptions import PermissionDenied

            raise PermissionDenied

        mesh = run.mesh

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        verboseid_s = _sanitize(mesh.verbose_id)[:50]

        static_models = os.path.join(settings.STATIC_ROOT, "models")

        pub_dir = os.path.join(static_models, str(mesh.ID), "published")

        # Only include published files for this specific run (workers copies as <meshID>_<runID>.<ext>)
        expected_prefix = f"{mesh.ID}_{run.ID}"
        files = []
        if os.path.isdir(pub_dir):
            for root, _, filenames in os.walk(pub_dir):
                for f in filenames:
                    if f.startswith(expected_prefix):
                        files.append(os.path.join(root, f))

        if not files:
            self.message_user(
                request,
                "No final published output found for this run.",
                level=messages.WARNING,
            )
            raise Http404("Published output not found")

        fname = f"{verboseid_s}_{str(run.ID)[:8]}.zip"

        # Write published files to a temp ZIP and stream after it's complete
        file_tuples = []
        for fp in files:
            try:
                arcname = os.path.relpath(fp, static_models)
            except Exception:
                arcname = os.path.basename(fp)
            file_tuples.append((fp, arcname))

        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()

        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for src, arc in file_tuples:
                        try:
                            zf.write(src, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {src} to published zip for run {run.ID}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating published zip for run {run.ID}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        writer_done.wait()

        def _stream():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(_stream(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            logging.info(
                f"ADMIN RUN PUBLISHED DOWNLOAD by {request.user} - run={run.ID} mesh={verboseid_s} filename={fname}"
            )
        except Exception:
            pass

        return resp

    def download_run_full_view(self, request, object_id):
        # Download the entire run directory (archived rundir)
        try:
            run = Run.objects.get(ID=object_id)
        except Run.DoesNotExist:
            raise Http404("Run not found")

        mesh = run.mesh

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        verboseid_s = _sanitize(mesh.verbose_id)[:50]
        # Determine run directory: archived runs are stored under ARCHIVE_ROOT, others under STATIC_ROOT/models
        if run.directory:
            run_dir_path = str(run.directory)
            # If stored as absolute path, use it directly
            if os.path.isabs(run_dir_path):
                run_dir = run_dir_path
            else:
                if run.status == "Archived":
                    base = settings.ARCHIVE_ROOT
                else:
                    base = os.path.join(settings.STATIC_ROOT, "models")
                run_dir = os.path.join(base, run_dir_path)
        else:
            run_dir = None

        files = []
        if run_dir and os.path.isdir(run_dir):
            for root, _, filenames in os.walk(run_dir):
                for f in filenames:
                    files.append(os.path.join(root, f))
        else:
            self.message_user(
                request, "Run directory not found for this run.", level=messages.WARNING
            )
            raise Http404("Run directory not found")

        fname = f"{verboseid_s}_full_{str(run.ID)[:8]}.zip"

        # Prepare list of files and arcname pairs to write into ZIP
        file_tuples = []
        for fp in files:
            try:
                arcname = os.path.relpath(fp, run_dir)
            except Exception:
                arcname = os.path.basename(fp)
            file_tuples.append((fp, arcname))

        # Stream ZIP by writing to a temp file in a background thread and reading while writing
        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()

        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp, arc in file_tuples:
                        try:
                            zf.write(fp, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {fp} to zip for run {run.ID}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating zip for run {run.ID}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()

        # Wait until writer finishes so client receives a complete ZIP
        writer_done.wait()

        def stream_file():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(stream_file(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            logging.info(
                f"ADMIN RUN FULL DOWNLOAD by {request.user} - run={run.ID} mesh={verboseid_s} filename={fname}"
            )
        except Exception:
            pass

        return resp

    def download_link(self, obj):
        try:
            url1 = reverse("admin:tirtha_run_download_output", args=[obj.ID])
            url2 = reverse("admin:tirtha_run_download_full", args=[obj.ID])
            return mark_safe(
                f'<a class="button" href="{url1}">Download Output</a>&nbsp;'
                f'<a class="button" href="{url2}">Download Full Run</a>'
            )
        except Exception:
            return ""

    download_link.short_description = "Download Output"

    def run_links(self, obj):
        """
        Render links for Runs.

        - "Open on Site": shown for `GS` and `aV` runs; links to the Run (via `BASE_URL/<ark>`).
        - "Edit in SuperSplat": shown only for `GS` runs; links to the .splat file.
        Returns "N.A." for other run kinds.
        """
        try:
            kind = getattr(obj, "kind", None)
            if kind not in ("GS", "aV"):
                return "N.A."

            mesh = obj.mesh
            # Use BASE_URL from settings as site base
            base = (
                settings.BASE_URL.rstrip("/")
                if getattr(settings, "BASE_URL", None)
                else ""
            )

            # Build run URL using the run's ARK
            run_ark = getattr(obj, "ark", None)
            run_url = (
                f"{base}/{run_ark}"
                if (base and run_ark)
                else (f"/{run_ark}" if run_ark else "")
            )

            links = []
            if run_url:
                links.append(
                    f'<a class="button" href="{run_url}" target="_blank" rel="noopener">Open on Site</a>'
                )

            # Add SuperSplat edit link only for GS runs
            if kind == "GS":
                ext = ".splat"
                filename = f"{mesh.ID}_{obj.ID}{ext}"
                splat_url = (
                    f"{base}/static/models/{mesh.ID}/published/{filename}"
                    if base
                    else f"/static/models/{mesh.ID}/published/{filename}"
                )
                ss_url = f"https://superspl.at/editor?load={splat_url}"
                links.append(
                    f'<a class="button" href="{ss_url}" target="_blank" rel="noopener">Edit in SuperSplat</a>'
                )

            if not links:
                return ""
            return mark_safe("&nbsp;".join(links))
        except Exception:
            return ""

    run_links.short_description = "View / Edit Output"

    # Replacement is handled via the change form
    def change_view(self, request, object_id, form_url="", extra_context=None):
        # If a replacement file was submitted via the change form, delegate
        # to the existing replace_output_view which performs permission checks
        # and the atomic replacement logic.
        try:
            if request.method == "POST" and request.FILES.get("replacement"):
                return self.replace_output_view(request, object_id)
        except Exception:
            # Let replace_output_view handle messaging/logging; fall through to normal handling
            pass

        return super().change_view(request, object_id, form_url, extra_context)

    def get_fieldsets(self, request, obj=None):
        """Return fieldsets but hide the `replacement` input unless the Run is Archived."""
        try:
            # Work on a shallow copy of the configured fieldsets
            fsets = [
                (title, dict(options))
                if isinstance(options, dict)
                else (title, options)
                for (title, options) in self.fieldsets
            ]
            if obj is None or getattr(obj, "status", None) != "Archived":
                # Modify the first fieldset's fields tuple to remove 'replacement' where present
                title, opts = fsets[0]
                fields = list(opts.get("fields", ()))
                if fields:
                    first = list(fields[0])
                    if "replacement" in first:
                        first = [f for f in first if f != "replacement"]
                        fields[0] = tuple(first)
                        opts["fields"] = tuple(fields)
                        fsets[0] = (title, opts)
            # Return as tuple in the same structure
            return tuple(fsets)
        except Exception:
            return self.fieldsets

    @admin.action(
        description="Download final published outputs (compressed) for selected runs"
    )
    def download_final_outputs(self, request, queryset):
        # Zips published output folders for selected runs (same logic as single-run published download)
        runs = list(queryset)
        if not runs:
            self.message_user(request, "No runs selected.", level=messages.WARNING)
            return

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        static_models = os.path.join(settings.STATIC_ROOT, "models")
        mesh_ids = {str(r.mesh.ID): r.mesh for r in runs}
        if len(mesh_ids) == 1:
            verboseid = _sanitize(next(iter(mesh_ids.values())).verbose_id)[:50]
        else:
            verboseid = "multiple"

        if len(runs) == 1:
            fname = f"{verboseid}_{str(runs[0].ID)[:8]}.zip"
        else:
            ids_short = "_".join([str(r.ID)[:4] for r in runs])
            fname = f"{verboseid}_{ids_short}.zip"

        # Collect published files for selected runs and write to temp ZIP
        file_tuples = []
        for run in runs:
            pub_dir = os.path.join(static_models, str(run.mesh.ID), "published")
            if not os.path.isdir(pub_dir):
                continue
            expected_prefix = f"{run.mesh.ID}_{run.ID}"
            for root, _, filenames in os.walk(pub_dir):
                for f in filenames:
                    if not f.startswith(expected_prefix):
                        continue
                    fp = os.path.join(root, f)
                    try:
                        rel = os.path.relpath(fp, static_models)
                        if len(mesh_ids) > 1:
                            mesh_vid = _sanitize(run.mesh.verbose_id)[:50]
                            arcname = os.path.join(mesh_vid, str(run.ID), rel)
                        else:
                            arcname = os.path.join(str(run.ID), rel)
                    except Exception:
                        arcname = os.path.join(str(run.ID), os.path.basename(fp))
                    file_tuples.append((fp, arcname))

        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()
        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for src, arc in file_tuples:
                        try:
                            zf.write(src, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {src} to published bulk zip {fname}"
                            )
                writer_done.set()
            except Exception:
                logging.exception(f"Error creating published bulk zip {fname}")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        writer_done.wait()

        def _stream():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(_stream(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            run_ids = [str(r.ID) for r in runs]
            logging.info(
                f"ADMIN RUNS PUBLISHED DOWNLOAD by {request.user} - runs={run_ids} filename={fname}"
            )
        except Exception:
            pass

        return resp

    def replace_output_view(self, request, object_id):
        """
        Admin view to replace the final published output file for a Run.
        Strictly enforces allowed upload types: .gltf/.glb for MeshOps and .splat for GSOps.
        Replacement is performed by writing the upload to a temp file, moving the original
        file to a timestamped .bak, then atomically moving the temp file to the original filename.
        Run.notes is updated only on success.
        # NOTE: Only staff users may perform this action.

        """
        from django.core.exceptions import PermissionDenied

        if not (request.user.is_staff or request.user.is_superuser):
            self.message_user(
                request,
                "Permission denied: staff or superuser only.",
                level=messages.ERROR,
            )
            raise PermissionDenied

        try:
            run = Run.objects.get(ID=object_id)
        except Run.DoesNotExist:
            raise Http404("Run not found")

        mesh = run.mesh
        static_models = os.path.join(settings.STATIC_ROOT, "models")
        pub_dir = os.path.join(static_models, str(mesh.ID), "published")

        # Find existing final output file for this run
        existing_file = None
        if os.path.isdir(pub_dir):
            # Prefer exact filename with expected extension (workers.py writes .glb for aV, .splat for GS)
            base_name = f"{mesh.ID}_{run.ID}"
            candidates = []
            if run.kind == "aV":
                candidates = [f"{base_name}.glb", f"{base_name}.gltf"]
            else:
                candidates = [f"{base_name}.splat"]

            for cand in candidates:
                cand_path = os.path.join(pub_dir, cand)
                if os.path.isfile(cand_path):
                    existing_file = cand
                    break

        if not existing_file:
            self.message_user(
                request,
                "Published final output not found for this run.",
                level=messages.ERROR,
            )
            raise Http404("Published output not found")

        if request.method == "GET":
            form_html = (
                f"<h2>Replace final output for Run {run.ID}</h2>"
                f"<p>Existing file: <strong>{existing_file}</strong></p>"
                '<form method="post" enctype="multipart/form-data">'
                "%s"
                '<input type="file" name="replacement" required>'
                '<input type="submit" value="Upload & Replace">'
                "</form>"
            )
            from django.middleware.csrf import get_token

            csrf = f'<input type="hidden" name="csrfmiddlewaretoken" value="{get_token(request)}">'
            return HttpResponse(form_html % csrf)

        # Process uploaded file
        uploaded = request.FILES.get("replacement")
        if not uploaded:
            self.message_user(request, "No file uploaded.", level=messages.ERROR)
            return HttpResponse(status=400)

        fname = uploaded.name
        lower = fname.lower()
        kind = run.kind
        # Strict file type enforcement
        if kind == "aV":
            if not (lower.endswith(".glb") or lower.endswith(".gltf")):
                self.message_user(
                    request,
                    "For MeshOps runs, only .glb/.gltf uploads are allowed.",
                    level=messages.ERROR,
                )
                return HttpResponse(status=400)
        else:
            if not lower.endswith(".splat"):
                self.message_user(
                    request,
                    "For GSOps runs, only .splat uploads are allowed.",
                    level=messages.ERROR,
                )
                return HttpResponse(status=400)

        orig_path = os.path.join(pub_dir, existing_file)
        if not os.path.isfile(orig_path):
            self.message_user(
                request, "Original final output file missing.", level=messages.ERROR
            )
            raise Http404("Original file missing")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        # Write uploaded file to a temp file in the same directory
        try:
            fd, tmp_path = tempfile.mkstemp(dir=pub_dir)
            with os.fdopen(fd, "wb") as tmpf:
                for chunk in uploaded.chunks():
                    tmpf.write(chunk)
                tmpf.flush()
                os.fsync(tmpf.fileno())
        except Exception:
            logging.exception(f"Error saving uploaded temp file for run {run.ID}")
            self.message_user(
                request, "Error saving uploaded file.", level=messages.ERROR
            )
            return HttpResponse(status=500)

        backup_name = f"{existing_file}.{timestamp}.bak"
        backup_path = os.path.join(pub_dir, backup_name)

        try:
            # Move original to backup, then atomically move temp to original path
            shutil.move(orig_path, backup_path)
            # Restrict backup file permissions to 0600
            try:
                os.chmod(backup_path, 0o600)
            except Exception:
                logging.exception(f"Failed to chmod backup file {backup_path}")

            os.replace(tmp_path, orig_path)
            # Ensure the replaced final output is world-readable/writeable by owner/group (0664)
            try:
                os.chmod(orig_path, 0o664)
            except Exception:
                logging.exception(f"Failed to chmod replaced file {orig_path}")
        except Exception:
            logging.exception(f"Error replacing final output for run {run.ID}")
            # Ensure temp file removed if exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            self.message_user(
                request,
                "Error performing replacement; original preserved as backup.",
                level=messages.ERROR,
            )
            return HttpResponse(status=500)

        # Update Run.notes and log the admin action
        try:
            note = f"Final output replaced by admin {request.user.username} at {datetime.now(timezone.utc).isoformat()} UTC. Original file moved to {backup_name}."
            if run.notes:
                run.notes = run.notes + "\n" + note
            else:
                run.notes = note
            run.save()
            logging.info(
                f"ADMIN RUN OUTPUT REPLACED by {request.user} - run={run.ID} original={existing_file} backup={backup_name} time={timestamp}"
            )
        except Exception:
            logging.exception(
                f"Error updating notes for run {run.ID} after replacement"
            )

        self.message_user(
            request, "Final output successfully replaced.", level=messages.SUCCESS
        )
        # Redirect back to change page
        change_url = reverse("admin:tirtha_run_change", args=[run.ID])

        return redirect(change_url)

    @admin.action(
        description="Download full run directories (compressed) for selected runs"
    )
    def download_full_runs(self, request, queryset):
        runs = list(queryset)
        if not runs:
            self.message_user(request, "No runs selected.", level=messages.WARNING)
            return

        def _sanitize(s: str) -> str:
            s = str(s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
            return s

        static_models = os.path.join(settings.STATIC_ROOT, "models")
        mesh_ids = {str(r.mesh.ID): r.mesh for r in runs}
        if len(mesh_ids) == 1:
            verboseid = _sanitize(next(iter(mesh_ids.values())).verbose_id)[:50]
        else:
            verboseid = "multiple"

        if len(runs) == 1:
            fname = f"{verboseid}_full_{str(runs[0].ID)[:8]}.zip"
        else:
            ids_short = "_".join([str(r.ID)[:4] for r in runs])
            fname = f"{verboseid}_full_{ids_short}.zip"

        # Build list of (fp, arcname) tuples for all runs, resolving archives appropriately
        file_tuples = []
        for run in runs:
            # Resolve run_dir similar to single-run logic
            if run.directory:
                run_dir_path = str(run.directory)
                if os.path.isabs(run_dir_path):
                    run_dir = run_dir_path
                else:
                    if run.status == "Archived":
                        base = settings.ARCHIVE_ROOT
                    else:
                        base = os.path.join(settings.STATIC_ROOT, "models")
                    run_dir = os.path.join(base, run_dir_path)
            else:
                run_dir = None

            if not run_dir or not os.path.isdir(run_dir):
                continue

            try:
                for root, _, filenames in os.walk(run_dir):
                    for f in filenames:
                        fp = os.path.join(root, f)
                        try:
                            if run.status == "Archived":
                                rel = os.path.relpath(fp, run_dir)
                            else:
                                rel = os.path.relpath(fp, static_models)

                            if len(mesh_ids) > 1:
                                mesh_vid = _sanitize(run.mesh.verbose_id)[:50]
                                arcname = os.path.join(mesh_vid, str(run.ID), rel)
                            else:
                                arcname = os.path.join(str(run.ID), rel)
                        except Exception:
                            arcname = os.path.join(str(run.ID), os.path.basename(fp))
                        file_tuples.append((fp, arcname))
            except Exception:
                logging.exception(
                    f"Error collecting files for run {run.ID} in bulk full-run action"
                )
                continue

        # Stream ZIP by writing to a temp file in a background thread
        tempf = tempfile.NamedTemporaryFile(delete=False)
        tempf_name = tempf.name
        tempf.close()

        writer_done = threading.Event()

        def _writer():
            try:
                with zipfile.ZipFile(tempf_name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp, arc in file_tuples:
                        try:
                            zf.write(fp, arc)
                        except Exception:
                            logging.exception(
                                f"Error adding file {fp} to zip in bulk full-run action"
                            )
                writer_done.set()
            except Exception:
                logging.exception("Error creating bulk full-runs zip")
                writer_done.set()

        t = threading.Thread(target=_writer, daemon=True)
        t.start()

        def stream_file():
            try:
                with open(tempf_name, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if chunk:
                            yield chunk
                        else:
                            if writer_done.is_set():
                                break
                            time.sleep(0.1)
            finally:
                try:
                    os.remove(tempf_name)
                except Exception:
                    pass

        resp = StreamingHttpResponse(stream_file(), content_type="application/zip")
        resp["Content-Disposition"] = f"attachment; filename={fname}"

        try:
            run_ids = [str(r.ID) for r in runs]
            logging.info(
                f"ADMIN RUNS FULL DOWNLOAD by {request.user} - runs={run_ids} filename={fname}"
            )
        except Exception:
            pass

        return resp

    actions = ["download_final_outputs", "download_full_runs"]


@admin.register(ARK)
class ARKAdmin(admin.ModelAdmin):
    def mesh_id_verbose(self, obj):
        return obj.run.mesh.verbose_id

    mesh_id_verbose.short_description = "Mesh ID (Verbose)"

    def get_run(self, obj):
        return obj.run

    get_run.short_description = "Run"

    def image_count(self, obj):
        return obj.run.images.count()

    image_count.short_description = "Total Image Count"

    readonly_fields = (
        "ark",
        "get_run",
        "mesh_id_verbose",
        "image_count",
        "naan",
        "shoulder",
        "assigned_name",
        "created_at",
        "url",
        "metadata",
    )
    fieldsets = (
        (
            "ARK Details",
            {
                "fields": (
                    "ark",
                    "url",
                    "created_at",
                    "get_run",
                    "mesh_id_verbose",
                    "image_count",
                    ("naan", "shoulder", "assigned_name"),
                    "metadata",
                    "commitment",
                )
            },
        ),
    )
    list_display = ("ark", "mesh_id_verbose", "get_run", "created_at", "image_count")
    list_per_page = 50
