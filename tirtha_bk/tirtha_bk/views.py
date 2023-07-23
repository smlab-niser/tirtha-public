from django.shortcuts import render

"""
Custom error pages

"""
def handler403(request, exception=None):
    """
    Permission denied.

    """
    return render(request, "tirtha/403.html", status=403)

def handler404(request, exception=None):
    """
    Page not found.

    """
    return render(request, "tirtha/404.html", status=404)

def handler500(request, exception=None):
    """
    Server error.

    """
    return render(request, "tirtha/500.html", status=500)

def handler503(request, exception=None):
    """
    Down for maintenance.

    """
    return render(request, "tirtha/503.html", status=503)
