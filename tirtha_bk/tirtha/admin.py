# import logging
from django.conf import settings
from django.contrib import admin, messages
from django.urls import reverse
from django.utils.html import mark_safe
from django.utils.translation import ngettext
from loguru import logger

# Local imports
from .models import ARK, Contribution, Contributor, Image, Mesh, Run
from .tasks import post_save_contrib_imageops, recon_runner_task


# Logger setup
# Logging
ADMIN_LOG_LOCATION = settings.ADMIN_LOG_LOCATION

# logging.basicConfig(
#     level=logging.NOTSET,
#     format="%(asctime)s %(levelname)s %(message)s",
#     filename=ADMIN_LOG_LOCATION,
# )
logger.remove()  # Remove default logger
logger.add(
    ADMIN_LOG_LOCATION,
    rotation="100 MB",
    retention="90 days",
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

    @admin.action(description="Mark selected meshes as completed")
    def mark_completed(self, request, queryset):
        updated = queryset.update(completed=True)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as completed.",
                "%d meshes were successfully marked as completed.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected meshes as incomplete")
    def mark_incomplete(self, request, queryset):
        updated = queryset.update(completed=False)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as incomplete.",
                "%d meshes were successfully marked as incomplete.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected meshes as hidden")
    def mark_hidden(self, request, queryset):
        updated = queryset.update(hidden=True)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as hidden.",
                "%d meshes were successfully marked as hidden.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected meshes as not hidden")
    def mark_not_hidden(self, request, queryset):
        updated = queryset.update(hidden=False)
        self.message_user(
            request,
            ngettext(
                "%d mesh was successfully marked as not hidden.",
                "%d meshes were successfully marked as not hidden.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    actions = [mark_completed, mark_incomplete, mark_hidden, mark_not_hidden]
    fieldsets = (
        (
            "Mesh Details",
            {
                "fields": (
                    ("ID", "verbose_id"),
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
    readonly_fields = (
        "ID",
        "verbose_id",
        "created_at",
        "updated_at",
        "reconstructed_at",
        "get_preview",
        "get_thumbnail",
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
    )
    list_per_page = 25
    inlines = [ContributionInlineMesh]  # , RunInlineMesh FIXME: Error while saving


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
            logger.info(
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
            logger.info(f"ADMIN -- aVOps successfully triggered for {obj.ID}.")
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
            logger.info(f"ADMIN -- GSOps successfully triggered for {obj.ID}.")
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

    actions = [mark_processed, trigger_imageops, trigger_aVOps, trigger_GSOps]
    readonly_fields = (
        "ID",
        "mesh",
        "contributor",
        "contributed_at",
        # "processed",
        "processed_at",
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
    )
    list_per_page = 50
    # inlines = [
    #     ImageInlineContribution,
    # ]


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


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    def mesh_id_verbose(self, obj):
        return obj.mesh.verbose_id

    mesh_id_verbose.short_description = "Mesh ID (Verbose)"

    def image_count(self, obj):
        return obj.images.count()

    image_count.short_description = "Image Count"

    readonly_fields = (
        "ID",
        "ark",
        "mesh_id_verbose",
        "kind",
        "started_at",
        "ended_at",
        "image_count",
        "status",
        "directory",
    )
    fieldsets = (
        (
            "Run Details",
            {
                "fields": (
                    ("ID"),
                    ("ark"),
                    ("mesh_id_verbose"),
                    ("kind"),
                    ("status"),
                    ("started_at", "ended_at"),
                    ("directory"),
                    ("image_count"),
                    (
                        "rotaZ",
                        "rotaX",
                        "rotaY",
                    ),  # Used only for <model-viewer>'s orientation
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
    )
    list_per_page = 50
    inlines = [
        ContributorInlineRun
    ]  # ImageInlineRun, FIXME: Too many images lead to 400 (Bad Request)


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
                    ("ark"),
                    ("url"),
                    ("created_at"),
                    ("get_run"),
                    ("mesh_id_verbose"),
                    ("image_count"),
                    ("naan", "shoulder", "assigned_name"),
                    ("metadata"),
                    ("commitment"),
                )
            },
        ),
    )
    list_display = ("ark", "mesh_id_verbose", "get_run", "created_at", "image_count")
    list_per_page = 50
