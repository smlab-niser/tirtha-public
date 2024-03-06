# NOTE: Rename this file to gunicorn.conf.py after editing
import os

workers = 16  # CHANGEME: as needed
syslog = True
bind = ["0.0.0.0:80"]  # CHANGEME:
umask = 0
loglevel = "info"
user = os.getenv("USER", "")  # CHANGEME:
group = os.getenv("USER", "")  # CHANGEME:
accesslog = "/var/www/tirtha/prod/logs/docker_gunicorn_access.log"  # CHANGEME: Use the value of LOG_DIR from local_settings.py with a filename, e.g., <LOG_DIR>/gunicorn_access.log
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" "%({X-Real-IP}i)s"'
)
errorlog = "/var/www/tirtha/prod/logs/docker_gunicorn_error.log"  # CHANGEME: Use the value of LOG_DIR from local_settings.py with a filename, e.g., <LOG_DIR>/gunicorn_error.log
