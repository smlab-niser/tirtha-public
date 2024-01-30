# NOTE: Rename this file to gunicorn.conf.py after editing
workers = 16  # CHANGEME: as needed
syslog = True
bind = ["<your_domain:your_port>"]  # e.g., tirtha.niser.ac.in:80  localhost;  # CHANGEME:
umask = 0
loglevel = "info"
user = "<your_user>"  # CHANGEME:
group = "<your_group>"  # CHANGEME:
accesslog = "<your_access_log_path>"  # CHANGEME: Use the value of LOG_DIR from local_settings.py with a filename, e.g., <LOG_DIR>/gunicorn_access.log
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" "%({X-Real-IP}i)s"'
)
errorlog = "<your_error_log_path>"  # CHANGEME: Use the value of LOG_DIR from local_settings.py with a filename, e.g., <LOG_DIR>/gunicorn_error.log
