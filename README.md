<h1 align="center">
    <img src="https://github.com/smlab-niser/tirtha-public/blob/dev/tirtha_bk/static/media/tirtha_logo.png" width=27 height=27>
    $\color{#ff4c40}{\textrm{Project Tirtha [Beta]}}$
</h1>

[Project Tirtha](https://tirtha.niser.ac.in) is an *academic initiative* to create
3D models of heritage sites using crowdsourced images. The word *Tirtha* is Sanskrit
for "a place of pilgrimage", and is commonly used to refer to the sacred sites of
Hinduism. Our goal is to preserve and showcase the beauty and cultural significance
of heritage sites. We believe that by allowing the general public to contribute to
the creation of these models, and by providing open access to these models, we can
increase awareness and appreciation of these important cultural landmarks and inspire
future generations to maintain them for years to come.

This project is now open-source under the [GNU Affero General Public License v3.0](./LICENSE)
and is under active development. All contributions are welcome. Please read
[CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## Requirements
### Hardware
* OS: Linux; Developed on Ubuntu 22.04 LTS
* RAM: 16 GB+
* VRAM: 6 GB+; NVIDIA GPU required for CUDA support

### System packages
* [Meshroom 2021.1.0](https://www.fosshub.com/Meshroom-old.html)
    * Download the archive from the releases page as linked above and extract it:
        ```sh
        tar -xvf Meshroom-2021.1.0-linux-cuda10.tar.gz
        ```
    * `cd` to the extracted directory - `cd ./Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/aliceVision/`.
    * Copy the compiled binaries and libraries from `./bin/aliceVision/bin/` & `./bin/aliceVision/lib/` and place them in the `bin21/` directory. Also, copy `cameraSensors.db` & `vlfeat_K80L3.SIFT.tree` from `./share/aliceVision/` to `bin21/`. See [below](#python) for the directory structure.
    * NOTE: We will upgrade to Meshroom 2023.x.x soon.
* [PostgreSQL](https://www.postgresql.org/download/)
* [Nginx](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-22-04)
* [RabbitMQ](https://www.rabbitmq.com/download.html) (for Celery)
* [`obj2gltf`](https://github.com/CesiumGS/obj2gltf)
    * Install using `npm install -g obj2gltf`.
* [`gltfpack`](https://github.com/zeux/meshoptimizer/) (for `meshoptimizer`).
    * Do not install via `npm`. Instead, download the binary from the releases page as linked [here](https://github.com/zeux/meshoptimizer/releases).
    * Add `gltfpack` to your system `PATH`.

### Python
* NN models
    * Create a `nn_models` directory under `tirtha_bk` (after you have cloned this repository) and clone [`nsfw-detector`](https://github.com/GantMan/nsfw_model) & [`MANIQA`](https://github.com/IIGROUP/MANIQA) there.
    * Download the weights for `nsfw-detector` from [here](https://github.com/GantMan/nsfw_model/releases/download/1.2.0/mobilenet_v2_140_224.1.zip) and unzip it in the `nsfw_model` directory.
    * `cd` to `nsfw-detector` and install it via `pip install -e .`.
    * `MANIQA` does not need to be installed. Just make sure that the `MANIQA` directory is present under `nn_models`.
    * Download the weights for `MANIQA` from [here](https://github.com/IIGROUP/MANIQA/releases/download/Kadid10k/ckpt_kadid10k.pt) and place it in the `MANIQA` directory.
    * The final directory structure should look like this:
        ```sh
        tirtha-public
        ├── <virtualenv>
        ├── <other_dirs>
        ├── tirtha_bk
            ├── config # <-- Config files for nginx & gunicorn placed here
            ├── static
            ├── tirtha
            ├── tirtha_bk
            ├── bin21 # <-- Meshroom binaries & libraries placed here
                ├── <aliceVision_* binaries>
                ├── <lib*.so* libraries>
                ├── cameraSensors.db
                ├── vlfeat_K80L3.SIFT.tree
            ├── nn_models # <-- NN models
                ├── MANIQA
                    ├── ckpt_kadid10k.pt # <-- Weights for MANIQA placed here
                ├── nsfw-detector
                    ├── mobilenet_v2_140_224 # <-- Weights for nsfw-detector unzipped here
        ```
* NOTE: `protobuf==3.20.3` is required by `nsfw-detector`.
* For other Python requirements, see [`requirements.txt`](./requirements.txt).

## Deployment / Development Setup

### Basic Setup
- Clone the repository and `cd` to it.
- Create a virtual environment and activate it.
- Install the system packages as listed under [`Requirements` > `System Packages`](#system-packages).
- Install the Python requirements via `pip install -r requirements.txt`. Note that `protobuf==3.20.3` is required by `nsfw-detector`. So, install it separately if you face any issues.
- For testing purposes, you can use SQLite as the database. For production, you will need to use PostgreSQL. Consult [here](https://www.digitalocean.com/community/tutorials/how-to-set-up-django-with-postgres-nginx-and-gunicorn-on-ubuntu-22-04) to set up `postgres`, `nginx` and `gunicorn`. Sample configuration files for `nginx` and `gunicorn` are provided in `tirtha_bk/config/`.
- A sample `local_settings.example.py` file is provided. Rename it to `local_settings.py` and edit it as required.
- Run `python manage.py makemigrations` and `python manage.py migrate` to create the database.
- Run `python manage.py runserver` to start the server. Or, use `gunicorn` as described [here](https://www.digitalocean.com/community/tutorials/how-to-set-up-django-with-postgres-nginx-and-gunicorn-on-ubuntu-22-04).
- Open `localhost:8000` in your browser to view the website.
- To access the admin panel, create a superuser using `python manage.py createsuperuser` and log in at `localhost:8000/admin`.

### Celery + RabbitMQ
- If you want to develop & test the backend, you will need to configure RabbitMQ and start the RabbitMQ server. Assuming that you have installed RabbitMQ, you can configure and start it as follows (on Debian / Ubuntu):
    ```sh
    # Initial setup
    sudo systemctl enable rabbitmq-server # For auto start on boot
    sudo systemctl start rabbitmq-server
    sudo rabbitmq-plugins enable rabbitmq_management # Accessible at http://localhost:15672/#/.

    # Add user, vhost and permissions - Same to be used in `local_settings.py`
    sudo rabbitmqctl add_user <username> <password> # CHANGEME:
    sudo rabbitmqctl add_vhost tirthamq
    sudo rabbitmqctl set_user_tags <username> administrator # CHANGEME:
    sudo rabbitmqctl set_permissions -p tirthamq <username> ".*" ".*" ".*" # For `tirthamq` vhost

    # Set consumer timeout to 1 year to avoid `ConnectionResetError: [Errno 104] Connection reset by peer` error.
    sudo rabbitmqctl eval 'application:get_env(rabbit, consumer_timeout).' # Check current timeout
    echo "consumer_timeout = 31556952000" | sudo tee -a /etc/rabbitmq/rabbitmq.conf
    sudo systemctl restart rabbitmq-server
    ```
- Next, you will need to start the Celery worker and beat scheduler via:
    ```sh
    celery -A tirtha worker -l INFO --max-tasks-per-child=1 -P threads --beat
    ```
- Here, `--max-tasks-per-child=1` is used to avoid high memory consumption; `--beat` is used to start the beat scheduler; and `-P threads` is used to use threads instead of processes, in order to avoid conflicts with `multiprocessing`. You can also use the `-D` flag to run the worker in the background. For logging, you can use the `-f` flag to specify a log file. See `celery worker --help` for more details.

## Acknowledgment
We thank the following individuals for their contributions to the project's development:
- [JeS24](https://github.com/JeS24)
- [annadapb](https://github.com/annadapb)

We are grateful to the developers of the following open-source libraries, which help make this project a reality:
- [AliceVision Meshroom](https://github.com/alicevision/Meshroom/), available under the [Mozilla Public License 2.0](https://github.com/alicevision/Meshroom/blob/develop/LICENSE-MPL2.md).
- [MANIQA](https://github.com/IIGROUP/MANIQA), available under the [Apache 2.0 License](https://github.com/IIGROUP/MANIQA/blob/master/LICENSE).
- [nsfw_model](https://github.com/GantMan/nsfw_model), available under the [MIT License](https://github.com/GantMan/nsfw_model/blob/master/LICENSE.md).
- [obj2gltf](https://github.com/CesiumGS/obj2gltf)
- [gltfpack](https://github.com/zeux/meshoptimizer)
- [model-viewer](https://github.com/google/model-viewer)
- [Google Fonts | Rubik](https://github.com/googlefonts/rubik)
- [Google Fonts | Material Icons](https://github.com/google/material-design-icons)
- [Django](https://github.com/django/django)
- [jQuery](https://github.com/jquery/jquery)
- [gunicorn](https://github.com/benoitc/gunicorn)
- [nginx](https://github.com/nginx/nginx)
- [Celery](https://github.com/celery/celery)
- [RabbitMQ](https://github.com/rabbitmq)
- [OpenCV](https://github.com/opencv/opencv)
- [Docker](https://github.com/docker)

We also thank Odisha State Archaeology for their support.

---
&copy; 2023 Project Tirtha,
[Subhankar Mishra's Lab](https://www.niser.ac.in/~smishra/),
[School of Computer Sciences](https://www.niser.ac.in/scps/), NISER.
All rights reserved.
