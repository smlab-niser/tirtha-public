# NOTE: Place this file in tirtha_bk/gunicorn/ and rename it to gunicorn.conf.py after editing
workers = 8  # CHANGEME: if needed
syslog = True
bind = ["<your_domain:your_port>"]  # e.g., tirtha.niser.ac.in:80 # CHANGEME:
umask = 0
loglevel = "info"
user = "<your_user>"  # CHANGEME:
group = "<your_group>"  # CHANGEME:
accesslog = "<your_access_log_path>"  # CHANGEME:
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" "%({X-Real-IP}i)s"'
)
errorlog = "<your_error_log_path>"  # CHANGEME:
