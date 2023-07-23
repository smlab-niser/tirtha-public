"""
This is intended to be used as a CLI tool
to get the verbose ID of a model (mesh), given its ID.

"""
import os

import django

# Setup Django pre-run to access the DB layer
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tirtha_bk.settings")
django.setup()

from tirtha.models import Mesh


def _get_mesh_details(meshID: str) -> dict:
    try:
        mesh = Mesh.objects.get(ID=meshID)
        return {
            "ID": mesh.ID,
            "verbose_id": mesh.verbose_id,
            "status": mesh.status,
            "completed": mesh.completed,
            "updated_at": mesh.updated_at.astimezone().strftime(
                "%d %b %Y, %H:%M:%S %Z"
            ),
            "reconstructed_at": mesh.reconstructed_at.astimezone().strftime(
                "%d %b %Y, %H:%M:%S %Z"
            )
            if mesh.reconstructed_at
            else None,
        }
    except Mesh.DoesNotExist as excep:
        return "Corresponding mesh not found."


if __name__ == "__main__":
    import argparse

    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser()
    parser.add_argument("meshID", help="Mesh ID")
    args = parser.parse_args()

    cons = Console()
    cons.rule("Mesh Details")

    dct = _get_mesh_details(args.meshID)

    tab = Table(show_header=False)
    for key, value in dct.items():
        tab.add_row(key, str(value))

    cons.print(tab)
    cons.rule()
