#!/bin/bash
# NOTE: Run as root (i.e., with sudo) - sudo bash build.sh, while in the `build/` folder.

# Some checks
# ==================================================================================================
# Exit on error
set -e

# Source the environment variables
source tirtha.env  # CHANGEME: NOTE: Edit the tirtha.env file to set the environment variables.

# Checking if the script is being run as root
if [ "$EUID" -ne 0 ]
  then echo "Please run the build.sh script as root (i.e., with sudo)."
  exit
fi

# Checking if the script is being run from the correct directory
if [ ! -f "build.sh" ]; then
  echo "Please run the build.sh script from the correct directory (i.e., ./tirtha-public/build/)."
  exit
fi

# Checking if the required files exist
# Checking if the tirtha.env file exists
if [ ! -f "tirtha.env" ]; then
  echo "Please make sure tirtha.env exists in ./tirtha-public/build/."
  exit
fi

# cd to project root, i.e., tirtha-public/
cd ../

# Checking if the requirements.txt file exists  # TODO: Add a frontend-only requirements file check here
if [ ! -f "./requirements.txt" ]; then
  echo "Please make sure requirements.txt exists in ./tirtha-public/."
  exit
fi

# Checking if the tirtha.env file has the required environment variables
if [ -z "$DB_NAME" ] || [ -z "$DB_USER" ] || [ -z "$DB_PWD" ] || [ -z "$RMQ_USER" ] || [ -z "$RMQ_PWD" ] || [ -z "$RMQ_VHOST" ] || [ -z "$DJANGO_SUPERUSER_NAME" ] || [ -z "$DJANGO_SUPERUSER_EMAIL" ] || [ -z "$DJANGO_SUPERUSER_PASSWORD" ] || [ -z "$GUNICORN_PORT" ]
  then echo "Please set the required environment variables in the tirtha.env file."
  exit
fi
# ==================================================================================================

# Main build process
# ==================================================================================================
# Installing dependencies
echo "Installing dependencies..."

apt-get update \
  && apt-get install -y \
    nano \
    curl \
    git \
    tmux \
    wget \
    unzip \
    libopencv-dev \
    nginx \
    rabbitmq-server \
    libpq-dev \
    postgresql \
    postgresql-contrib \
    python3.11-dev \
    python3.11-venv \
    python3-pip

# Getting submodules for ImageOps models
git config --global http.postBuffer 524288000 \
&& git submodule update --init --recursive  # CHANGEME: NOTE: Comment out to skip ImageOps models

# Creating a Python virtual environment and installing dependencies
python3.11 -m venv venv \
  && source ./venv/bin/activate \
  && pip install --upgrade pip setuptools wheel \
  && pip install -r ./requirements.txt --default-timeout=2000 \  # TODO: Add a frontend-only requirements file
  && pip install -e ./tirtha_bk/nn_models/nsfw_model/ \  # CHANGEME: NOTE: Comment out to skip one of ImageOps model
  && pip install protobuf==3.20.3  # CHANGEME: NOTE: Comment out to skip one of ImageOps model

# Getting the pre-trained checkpoints for ImageOps models
# CHANGEME: NOTE: Comment these out to skip the ImageOps models
wget https://smlab.niser.ac.in/project/tirtha/static/artifacts/MR2021.1.0.zip \
  && unzip MR2021.1.0.zip \
  && mv ./bin21/ ./tirtha_bk/bin21/ \
  && rm ./MR2021.1.0.zip
wget https://smlab.niser.ac.in/project/tirtha/static/artifacts/ckpt_kadid10k.pt \
  && mv ./ckpt_kadid10k.pt ./tirtha_bk/nn_models/MANIQA/

# Setting up npm to install obj2gltf and gltfpack
# CHANGEME: NOTE: Comment these out if developing only for the frontend
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
. /root/.bashrc
  && nvm install node \
  && nvm use node \
  && npm install -g obj2gltf
wget https://github.com/zeux/meshoptimizer/releases/download/v0.20/gltfpack-ubuntu.zip \
  && unzip gltfpack-ubuntu.zip \
  && chmod +x gltfpack \
  && mv gltfpack /usr/local/bin/ \
  && rm /gltfpack-ubuntu.zip

# Copying the configuration files to the appropriate locations
mv ./tirtha_bk/config/tirtha.docker.nginx ./tirtha_bk/config/tirtha.nginx \
  && mv ./tirtha_bk/tirtha_bk/local_settings.docker.py ./tirtha_bk/tirtha_bk/local_settings.py \
  && mv ./tirtha_bk/gunicorn/gunicorn.conf.docker.py ./tirtha_bk/gunicorn/gunicorn.conf.py

# Copying the production folder to the container and setting permissions
cp -r ./tirtha /var/www/tirtha
chmod -R 755 /var/www/tirtha/
chown -R $(whoami):$(whoami) /var/www/tirtha/
# ==================================================================================================

# Main Configuration
# ==================================================================================================
echo "Configuring dependencies and Tirtha..."

# Nginx
cp ./tirtha_bk/config/tirtha.nginx /etc/nginx/sites-available/tirtha
ln -s /etc/nginx/sites-available/tirtha /etc/nginx/sites-enabled/

# Postgres
source init_db.sh
systemctl start postgresql
systemctl enable postgresql

# RabbitMQ
systemctl start rabbitmq-server
systemctl enable rabbitmq-server
rabbitmq-plugins enable rabbitmq_management
rabbitmqctl add_user $RMQ_USER $RMQ_PWD
rabbitmqctl add_vhost $RMQ_VHOST
rabbitmqctl set_user_tags $RMQ_USER administrator
rabbitmqctl set_permissions -p $RMQ_VHOST $RMQ_USER ".*" ".*" ".*"
rabbitmqctl eval 'application:get_env(rabbit, consumer_timeout).'
echo "consumer_timeout = 31556952000" | tee -a /etc/rabbitmq/rabbitmq.conf
systemctl restart rabbitmq-server

# Copying error templates
cp tirtha-public/tirtha_bk/tirtha/templates/tirtha/403.html /var/www/tirtha/errors/
cp tirtha-public/tirtha_bk/tirtha/templates/tirtha/404.html /var/www/tirtha/errors/
cp tirtha-public/tirtha_bk/tirtha/templates/tirtha/500.html /var/www/tirtha/errors/
cp tirtha-public/tirtha_bk/tirtha/templates/tirtha/503.html /var/www/tirtha/errors/

# Django
python ./tirtha_bk/manage.py makemigrations tirtha
python ./tirtha_bk/manage.py collectstatic --no-input
python ./tirtha_bk/manage.py migrate
DJANGO_SUPERUSER_PASSWORD=$DJANGO_SUPERUSER_PASSWORD python ./tirtha_bk/manage.py createsuperuser --no-input --username $DJANGO_SUPERUSER_NAME --email "$DJANGO_SUPERUSER_EMAIL"
# ==================================================================================================

# Starting services
# ==================================================================================================
# Trigger the services
# echo "Starting services..."
# source start.sh  # CHANGEME: NOTE: Uncomment if you want to start the services after the build
echo "Run the start.sh script to start the services manually."
# ==================================================================================================
