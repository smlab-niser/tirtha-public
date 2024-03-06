#!/bin/bash
# NOTE: Run as root (i.e., with sudo) - sudo bash build.sh, while in the `build/` folder.

# Some checks
# ==================================================================================================
# Exit on error
set -e

# Checking if the script is being run from the correct directory
if [ "$(basename "$(pwd)")" != "build" ]; then
  echo "Please run the build.sh script from the correct directory (i.e., ./tirtha-public/build/)."
  exit
fi

# Source the environment variables
source tirtha.env  # CHANGEME: NOTE: Edit the tirtha.env file to set the environment variables.

# cd to project root, i.e., tirtha-public/
cd ../
# ==================================================================================================

# Triggering the services
# ==================================================================================================
# Activating the virtual environment
source ./venv/bin/activate

# cd to the backend directory
cd ./tirtha_bk/

# Starting celery in a tmux session
tmux new-session -d -s celery_session || tmux attach-session -t celery_session
tmux send-keys -t celery_session "celery -A tirtha worker -l INFO --max-tasks-per-child=1 -P threads --beat" C-m

# Starting the frontend | NOTE: Browse to HOST_IP:8000 in a browser to access the frontend.
gunicorn --bind 0.0.0.0:$GUNICORN_PORT tirtha_bk.wsgi
# ==================================================================================================
