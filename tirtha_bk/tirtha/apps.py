from django.apps import AppConfig


class TirthaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "tirtha"

    def ready(self):
        import tirtha.signals

        return super().ready()
